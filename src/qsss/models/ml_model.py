"""机器学习模型模块"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config.settings import settings

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = ""
except (ImportError, OSError) as exc:
    lgb = None  # type: ignore[assignment]
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = str(exc)


class MLModel:
    """机器学习模型类"""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.scaler: StandardScaler = StandardScaler()
        self.features = [
            "momentum_1m",
            "momentum_3m",
            "momentum_6m",
            "volatility",
            "vol_ratio",
            "rsi",
            "macd",
            "signal",
        ]

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备特征数据"""
        try:
            # 确保数据完整性
            df = df.copy()

            # 处理异常值
            for col in ["close", "volume", "turn"]:
                if col in df.columns:
                    median = df[col].median()
                    std = df[col].std()
                    df[col] = df[col].clip(median - 3 * std, median + 3 * std)

            # 确保没有零值和负值
            df["close"] = df["close"].replace(0, np.nan)
            df["volume"] = df["volume"].replace(0, np.nan)
            df["turn"] = df["turn"].replace(0, np.nan)

            # 计算特征
            df["momentum_1m"] = df["close"].pct_change(20)
            df["momentum_3m"] = df["close"].pct_change(60)
            df["momentum_6m"] = df["close"].pct_change(120)

            # 计算技术指标
            df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
            df["ma20"] = df["close"].rolling(window=20, min_periods=5).mean()

            volume_ma = df["volume"].rolling(window=20, min_periods=5).mean()
            df["vol_ratio"] = df["volume"] / volume_ma.replace(0, np.nan)
            df["vol_ratio"] = df["vol_ratio"].fillna(1.0).clip(0, 10)

            returns = df["close"].pct_change()
            df["volatility"] = returns.rolling(
                window=20, min_periods=5
            ).std() * np.sqrt(252)
            df["volatility"] = (
                df["volatility"].fillna(df["volatility"].mean()).clip(0, 2)
            )

            # RSI计算
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df["rsi"] = 100 - (100 / (1 + rs))
            df["rsi"] = df["rsi"].fillna(50).clip(0, 100)

            # MACD计算
            df["ema12"] = df["close"].ewm(span=12, adjust=False, min_periods=1).mean()
            df["ema26"] = df["close"].ewm(span=26, adjust=False, min_periods=1).mean()
            df["macd"] = df["ema12"] - df["ema26"]
            df["signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=1).mean()

            # 填充缺失值
            df = df.ffill().bfill()

            return df

        except Exception as e:
            logger.error(f"准备特征数据失败: {e}")
            return pd.DataFrame()

    def train_model(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[Any], Optional[StandardScaler]]:
        """训练机器学习模型"""
        try:
            if not LIGHTGBM_AVAILABLE or lgb is None:
                logger.error(f"LightGBM 运行库不可用: {LIGHTGBM_IMPORT_ERROR}")
                return None, None

            if len(df) < 100:  # 数据太少
                logger.warning(f"数据量不足: {len(df)} < 100")
                return None, None

            # 准备特征
            df = self.prepare_features(df)
            if df.empty:
                return None, None

            # 创建目标变量（5日收益率）
            df["target"] = df["close"].shift(-5) / df["close"] - 1

            # 只按训练所需列删除缺失值，避免非训练辅助列清空训练集。
            df = df.dropna(subset=[*self.features, "target"])
            if len(df) < 50:  # 训练数据不足
                return None, None

            # 准备训练数据
            X = df[self.features]
            y = (df["target"] > df["target"].mean()).astype(int)

            # 数据标准化
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.features,
                index=X.index,
            )

            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y,
                test_size=settings.ml_test_size,
                random_state=settings.ml_random_state,
            )

            # 训练模型
            self.model = lgb.LGBMClassifier(
                random_state=settings.ml_random_state,
                n_estimators=100,
                num_leaves=31,
                min_child_samples=5,
                max_depth=5,
                learning_rate=0.1,
                verbose=-1,
            )

            self.model.fit(X_train, y_train)

            # 评估模型
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)

            logger.info(
                f"模型训练完成 - 训练集准确率: {train_score:.3f}, 测试集准确率: {test_score:.3f}"
            )

            return self.model, self.scaler

        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            return None, None

    def predict(self, df: pd.DataFrame, model: Any, scaler: StandardScaler) -> float:
        """预测上涨概率"""
        try:
            df = self.prepare_features(df)
            if df.empty:
                return 0.5

            # 获取最新数据
            latest_data = df.iloc[-1:][self.features]
            if latest_data.isnull().any().any():
                return 0.5

            latest_scaled = pd.DataFrame(
                scaler.transform(latest_data),
                columns=self.features,
                index=latest_data.index,
            )
            prediction = model.predict_proba(latest_scaled)[0][1]

            return float(prediction)

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return 0.5
