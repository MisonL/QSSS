"""超短线策略分析模块"""

from typing import Any, Dict

import pandas as pd
from loguru import logger


class ShortTermAnalyzer:
    """超短线策略分析器"""

    def __init__(self) -> None:
        self.weights = {
            "volume_ratio": 0.3,
            "turnover_ratio": 0.2,
            "macd_cross": 0.2,
            "price_position": 0.15,
            "short_momentum": 0.15,
        }

    def calculate_explosion_score(self, df: pd.DataFrame) -> float:
        """计算超短线爆发潜力得分"""
        try:
            if len(df) < 20:
                return 0.0

            # 1. 成交量分析
            recent_volume_mean = df["volume"].iloc[-20:].mean()
            if recent_volume_mean == 0:
                volume_ratio = 0
            else:
                volume_ratio = df["volume"].iloc[-1] / recent_volume_mean
            volume_ratio = min(max(volume_ratio, 0), 5)

            # 2. 换手率分析
            recent_turn_mean = df["turn"].iloc[-20:].mean()
            if recent_turn_mean == 0:
                turnover_ratio = 0
            else:
                turnover_ratio = df["turn"].iloc[-1] / recent_turn_mean
            turnover_ratio = min(max(turnover_ratio, 0), 5)

            # 3. MACD金叉预判
            if len(df) >= 2:
                last_macd_diff = df["macd"].iloc[-1] - df["signal"].iloc[-1]
                prev_macd_diff = df["macd"].iloc[-2] - df["signal"].iloc[-2]
                macd_cross_score = (
                    1.0 if (last_macd_diff > 0 and prev_macd_diff < 0) else 0.0
                )
            else:
                macd_cross_score = 0.0

            # 4. 股价位置分析
            price_range = df["close"].iloc[-20:].max() - df["close"].iloc[-20:].min()
            if price_range > 0:
                price_position = (
                    df["close"].iloc[-1] - df["close"].iloc[-20:].min()
                ) / price_range
            else:
                price_position = 0.5

            # 5. 短期动量
            if len(df) >= 3:
                recent_momentum = df["close"].iloc[-1] / df["close"].iloc[-3] - 1
                recent_momentum = min(max(recent_momentum * 100, -20), 20)
                short_momentum = (recent_momentum + 20) / 40
            else:
                short_momentum = 0.5

            # 计算综合得分
            explosion_score = (
                volume_ratio * self.weights["volume_ratio"]
                + turnover_ratio * self.weights["turnover_ratio"]
                + macd_cross_score * self.weights["macd_cross"]
                + (1 - price_position) * self.weights["price_position"]
                + short_momentum * self.weights["short_momentum"]
            )

            return float(explosion_score)

        except Exception as e:
            logger.error(f"计算爆发潜力失败: {e}")
            return 0.0

    def get_short_term_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取超短线交易信号"""
        try:
            if len(df) < 10:
                return {}

            signals = {
                "explosion_score": self.calculate_explosion_score(df),
                "volume_surge": False,
                "turnover_surge": False,
                "price_breakout": False,
                "momentum_acceleration": False,
            }

            # 成交量激增信号
            if len(df) >= 20:
                recent_volume = df["volume"].iloc[-1]
                avg_volume = df["volume"].iloc[-20:-1].mean()
                if avg_volume > 0 and recent_volume / avg_volume > 2:
                    signals["volume_surge"] = True

            # 换手率激增信号
            if len(df) >= 20:
                recent_turn = df["turn"].iloc[-1]
                avg_turn = df["turn"].iloc[-20:-1].mean()
                if avg_turn > 0 and recent_turn / avg_turn > 1.5:
                    signals["turnover_surge"] = True

            # 价格突破信号
            if len(df) >= 5:
                current_price = df["close"].iloc[-1]
                ma5 = df["close"].iloc[-5:].mean()
                if current_price > ma5 * 1.02:
                    signals["price_breakout"] = True

            # 动量加速信号
            if len(df) >= 5:
                recent_return = df["close"].iloc[-1] / df["close"].iloc[-5] - 1
                if recent_return > 0.05:
                    signals["momentum_acceleration"] = True

            return signals

        except Exception as e:
            logger.error(f"获取超短线信号失败: {e}")
            return {}
