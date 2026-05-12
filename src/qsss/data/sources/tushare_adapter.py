"""Tushare 数据源适配器

该适配器实现了 DataAdapter 接口，作为 pytdx 之外的可选行情数据源。
默认不会启用，只有在安装了 `tushare` 并配置了环境变量
`QSSS_TUSHARE_TOKEN` 或 `TUSHARE_TOKEN` 时才会被 DataManager 注册。

注意：
- 需要自行在 Tushare 官网申请 Pro Token。
- 数据字段会被转换成与 PytdxAdapter 一致的结构，以便复用现有
  策略与特征工程逻辑。
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd
from loguru import logger

from ...config.settings import settings
from ...utils.decorators import retry_on_exception
from ..adapters import DataAdapter

try:  # 懒加载 tushare 依赖
    import tushare as ts

    TUSHARE_IMPORTED = True
except ImportError:  # pragma: no cover - 仅在未安装 tushare 时触发
    TUSHARE_IMPORTED = False
    logger.warning("tushare 未安装，如需使用 TushareAdapter 请先安装 tushare")


def _get_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """获取 Tushare Token，优先级：显式参数 > QSSS_TUSHARE_TOKEN > TUSHARE_TOKEN。"""
    if explicit_token:
        return explicit_token
    return os.getenv("QSSS_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")


def _symbol_to_ts_code(symbol: str) -> str:
    """将 6 位股票代码转换为 Tushare ts_code（如 000001.SZ）。"""
    symbol = str(symbol).strip()
    if not symbol or len(symbol) != 6:
        return symbol

    if symbol.startswith(("60", "68")):
        return f"{symbol}.SH"
    elif symbol.startswith(("00", "30")):
        return f"{symbol}.SZ"
    elif symbol.startswith(("83", "87", "43", "430", "889")):
        return f"{symbol}.BJ"
    # 其他情况默认深市
    return f"{symbol}.SZ"


class TushareAdapter(DataAdapter):
    """基于 Tushare Pro 的数据源适配器。"""

    _lock: threading.Lock
    _last_call_ts: float
    _min_interval: float

    def _throttle(self) -> None:
        """在多线程环境下对 Tushare 调用做简单节流，避免触发频率限制。"""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_call_ts
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
                now = time.monotonic()
            self._last_call_ts = now

    def __init__(self, token: Optional[str] = None) -> None:
        if not TUSHARE_IMPORTED:
            raise ImportError("tushare 库未安装，无法使用 TushareAdapter")

        token_value = _get_token(token)
        if not token_value:
            raise RuntimeError(
                "未找到 Tushare Token，请设置环境变量 QSSS_TUSHARE_TOKEN 或 TUSHARE_TOKEN"
            )

        # 初始化 Tushare Pro 客户端
        ts.set_token(token_value)
        self._pro = ts.pro_api(token_value)
        self._token = token_value

        # 简单节流控制，避免在多线程环境中过快打满 Tushare 限流
        self._lock = threading.Lock()
        self._last_call_ts = 0.0
        self._min_interval = float(settings.tushare_min_interval)

    # ------------------------------------------------------------------
    # 股票列表
    # ------------------------------------------------------------------
    @retry_on_exception(
        retries=settings.retry_count, delay=settings.retry_delay, backoff=2.0
    )
    def get_stock_list(self) -> pd.DataFrame:
        """获取 A 股列表，返回列：symbol, name, market。

        与 PytdxAdapter 一致：market 字段采用“交易所-板块”描述，如：
        - 上交所-主板
        - 上交所-科创板
        - 深交所-主板 / 中小板 / 创业板
        - 北交所-主板
        """
        try:
            self._throttle()
            df = self._pro.stock_basic(
                exchange="",
                list_status="L",  # 仅在市股票
                fields="ts_code,symbol,name,market,list_status",
            )
        except Exception as e:  # pragma: no cover - 依赖外部服务
            logger.error(f"Tushare 获取股票列表失败: {e}")
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # 过滤 ST / 退市标记（名称中通常包含 ST / 退）
        df = df[~df["name"].str.contains("ST") & ~df["name"].str.contains("退")].copy()

        def _map_market(row: pd.Series) -> str:
            sym = row["symbol"]
            mkt = (row.get("market") or "").strip()

            if mkt in {"主板", "MAIN"}:
                if sym.startswith("60"):
                    return "上交所-主板"
                elif sym.startswith("00"):
                    return "深交所-主板"
            if mkt in {"中小板"}:
                return "深交所-中小板"
            if mkt in {"创业板"}:
                return "深交所-创业板"
            if mkt in {"科创板"}:
                return "上交所-科创板"
            if mkt in {"北交所", "BSE"}:
                return "北交所-主板"
            return mkt or "其他"

        df["market"] = df.apply(_map_market, axis=1)
        return df[["symbol", "name", "market"]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 日线数据
    # ------------------------------------------------------------------
    @retry_on_exception(
        retries=settings.retry_count, delay=settings.retry_delay, backoff=2.0
    )
    def get_daily_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取日线数据，输出字段与 PytdxAdapter 保持一致。

        返回列至少包括：
        - date: YYYY-MM-DD
        - open, high, low, close
        - volume, amount
        - amplitude, pct_chg, change, turn
        """
        ts_code = _symbol_to_ts_code(symbol)

        # Tushare 使用 YYYYMMDD 格式
        sd = start_date or "20220101"
        ed = end_date or datetime.now().strftime("%Y%m%d")

        try:
            self._throttle()
            price = self._pro.daily(ts_code=ts_code, start_date=sd, end_date=ed)
        except Exception as e:  # pragma: no cover - 外部服务依赖
            logger.error(f"Tushare 获取 {ts_code} 日线数据失败: {e}")
            return pd.DataFrame()

        if price.empty:
            return pd.DataFrame()

        # 尝试获取换手率（非必需，获取失败时可回退为 0）
        try:
            self._throttle()
            basic = self._pro.daily_basic(
                ts_code=ts_code,
                start_date=sd,
                end_date=ed,
                fields="trade_date,turnover_rate",
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"Tushare 获取 {ts_code} daily_basic 失败: {e}")
            basic = pd.DataFrame()

        df = price.copy()
        if not basic.empty:
            df = df.merge(basic, on="trade_date", how="left")
        else:
            df["turnover_rate"] = 0.0

        # 统一字段
        df = df.sort_values("trade_date").reset_index(drop=True)

        # 日期格式转换
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        # 数值列
        df.rename(columns={"vol": "volume"}, inplace=True)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 涨跌幅、涨跌额、振幅（如果 Tushare 提供 pct_chg 则直接使用，否则本地计算）
        if "pct_chg" in df.columns:
            df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
        else:
            df["pct_chg"] = (df["close"] / df["close"].shift(1) - 1) * 100

        df["change"] = df["close"] - df["close"].shift(1)
        df["amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1) * 100

        # 换手率
        df["turn"] = pd.to_numeric(
            df.get("turnover_rate", 0.0), errors="coerce"
        ).fillna(0.0)

        # 清理数据
        df = df.dropna(subset=["open", "high", "low", "close"]).copy()

        return df[
            [
                "date",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "amplitude",
                "pct_chg",
                "change",
                "turn",
            ]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 实时数据
    # ------------------------------------------------------------------
    @retry_on_exception(
        retries=settings.retry_count, delay=settings.retry_delay, backoff=2.0
    )
    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时数据（基于 ts.get_realtime_quotes）。

        输出字段尽量与 PytdxAdapter 对齐：
        - symbol, name
        - price, last_close, open, high, low
        - volume, amount
        - time
        """
        if not symbols:
            return pd.DataFrame()

        codes = [str(s).strip() for s in symbols if str(s).strip()]
        if not codes:
            return pd.DataFrame()

        try:
            # 旧版 Tushare 实时行情接口
            self._throttle()
            df = ts.get_realtime_quotes(codes)
        except Exception as e:  # pragma: no cover - 依赖外部服务
            logger.error(f"Tushare 获取实时行情失败: {e}")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        # 统一代码格式
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)

        # 只保留请求的代码
        if "code" in df.columns:
            df = df[df["code"].isin(codes)]

        # 数值列转换
        for col in ["price", "pre_close", "open", "high", "low", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        out = pd.DataFrame()
        if "code" in df.columns:
            out["symbol"] = df["code"]
        if "name" in df.columns:
            out["name"] = df["name"]
        if "price" in df.columns:
            out["price"] = df["price"]
        if "pre_close" in df.columns:
            out["last_close"] = df["pre_close"]
        if "open" in df.columns:
            out["open"] = df["open"]
        if "high" in df.columns:
            out["high"] = df["high"]
        if "low" in df.columns:
            out["low"] = df["low"]
        if "volume" in df.columns:
            out["volume"] = df["volume"]
        if "amount" in df.columns:
            out["amount"] = df["amount"]
        if "time" in df.columns:
            out["time"] = df["time"]

        if out.empty:
            return pd.DataFrame()
        return out.reset_index(drop=True)
