"""技术指标计算模块"""

from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalAnalyzer:
    """技术指标分析器"""

    @staticmethod
    def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """计算动量因子"""
        df = df.copy()
        df["momentum_1m"] = df["close"].pct_change(20)  # 1个月动量
        df["momentum_3m"] = df["close"].pct_change(60)  # 3个月动量
        df["momentum_6m"] = df["close"].pct_change(120)  # 6个月动量
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).clip(0, 100)

    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        ema12 = df["close"].ewm(span=12, adjust=False, min_periods=1).mean()
        ema26 = df["close"].ewm(span=26, adjust=False, min_periods=1).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()

        return {"macd": macd, "signal": signal, "histogram": macd - signal}

    @staticmethod
    def get_macd_signal(df: pd.DataFrame) -> str:
        """根据MACD判断当前信号状态

        返回值示例："golden_cross"、"dead_cross"、"bullish"、"bearish"、"neutral"、"unknown"。
        """
        try:
            if len(df) < 2:
                return "unknown"

            if "macd" in df.columns and "signal" in df.columns:
                macd = df["macd"]
                signal = df["signal"]
            else:
                macd_data = TechnicalAnalyzer.calculate_macd(df)
                macd = macd_data["macd"]
                signal = macd_data["signal"]

            last_diff = macd.iloc[-1] - signal.iloc[-1]
            prev_diff = macd.iloc[-2] - signal.iloc[-2]

            if last_diff > 0 and prev_diff <= 0:
                return "golden_cross"
            if last_diff < 0 and prev_diff >= 0:
                return "dead_cross"
            if last_diff > 0:
                return "bullish"
            if last_diff < 0:
                return "bearish"
            return "neutral"

        except Exception as e:
            logger.error(f"计算MACD信号失败: {e}")
            return "unknown"

    @staticmethod
    def calculate_bollinger_bands(
        df: pd.DataFrame, period: int = 20
    ) -> Dict[str, pd.Series]:
        """计算布林带"""
        middle = df["close"].rolling(window=period, min_periods=5).mean()
        std = df["close"].rolling(window=period, min_periods=5).std()
        upper = middle + 2 * std
        lower = middle - 2 * std

        return {"boll_mid": middle, "boll_up": upper, "boll_down": lower}

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算波动率"""
        returns = df["close"].pct_change()
        volatility = returns.rolling(window=period, min_periods=5).std() * np.sqrt(252)
        return volatility.fillna(volatility.mean()).clip(0, 2)

    @staticmethod
    def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算成交量比率"""
        volume_ma = df["volume"].rolling(window=period, min_periods=5).mean()
        vol_ratio = df["volume"] / volume_ma.replace(0, np.nan)
        return vol_ratio.fillna(1.0).clip(0, 10)

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()

        # 基础数据清洗
        for col in ["close", "volume", "turn"]:
            if col in df.columns:
                median = df[col].median()
                std = df[col].std()
                df[col] = df[col].clip(median - 3 * std, median + 3 * std)

        # 计算动量
        df = TechnicalAnalyzer.calculate_momentum(df)

        # 计算RSI
        df["rsi"] = TechnicalAnalyzer.calculate_rsi(df)

        # 计算MACD
        macd_data = TechnicalAnalyzer.calculate_macd(df)
        df["macd"] = macd_data["macd"]
        df["signal"] = macd_data["signal"]

        # 计算布林带
        bb_data = TechnicalAnalyzer.calculate_bollinger_bands(df)
        df["boll_mid"] = bb_data["boll_mid"]
        df["boll_up"] = bb_data["boll_up"]
        df["boll_down"] = bb_data["boll_down"]

        # 计算波动率
        df["volatility"] = TechnicalAnalyzer.calculate_volatility(df)

        # 计算成交量比率
        df["vol_ratio"] = TechnicalAnalyzer.calculate_volume_ratio(df)

        # 填充缺失值
        df = df.ffill().bfill()

        return df

    @staticmethod
    def calculate_short_term_explosion(df: pd.DataFrame) -> float:
        """计算超短线爆发潜力"""
        try:
            if len(df) < 20:
                return 0.0

            # 1. 成交量分析
            recent_volume_mean = df["volume"].iloc[-20:].mean()
            volume_ratio = (
                df["volume"].iloc[-1] / recent_volume_mean
                if recent_volume_mean > 0
                else 0
            )

            # 2. 换手率分析
            recent_turn_mean = df["turn"].iloc[-20:].mean()
            turnover_ratio = (
                df["turn"].iloc[-1] / recent_turn_mean if recent_turn_mean > 0 else 0
            )

            # 3. MACD金叉预判
            macd_data = TechnicalAnalyzer.calculate_macd(df)
            last_macd_diff = macd_data["macd"].iloc[-1] - macd_data["signal"].iloc[-1]
            prev_macd_diff = macd_data["macd"].iloc[-2] - macd_data["signal"].iloc[-2]
            macd_cross_score = (
                1.0 if (last_macd_diff > 0 and prev_macd_diff < 0) else 0.0
            )

            # 4. 股价趋势分析
            price_range = df["close"].iloc[-20:].max() - df["close"].iloc[-20:].min()
            if price_range > 0:
                price_position = (
                    df["close"].iloc[-1] - df["close"].iloc[-20:].min()
                ) / price_range
            else:
                price_position = 0.5

            # 5. 短期动量加速
            recent_momentum = (
                df["close"].iloc[-1] / df["close"].iloc[-3] - 1
                if df["close"].iloc[-3] > 0
                else 0
            )

            # 标准化各个指标
            volume_ratio = min(max(volume_ratio, 0), 5)
            turnover_ratio = min(max(turnover_ratio, 0), 5)
            recent_momentum = min(max(recent_momentum * 100, -20), 20)

            # 计算综合得分
            explosion_score = (
                volume_ratio * 0.3
                + turnover_ratio * 0.2
                + macd_cross_score * 0.2
                + (1 - price_position) * 0.15
                + (recent_momentum + 20) / 40 * 0.15
            )

            return float(explosion_score)

        except Exception as e:
            logger.error(f"计算超短线爆发潜力失败: {e}")
            return 0.0
