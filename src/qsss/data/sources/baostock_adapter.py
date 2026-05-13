"""Baostock 数据源适配器

该适配器实现 DataAdapter 接口，作为另一种可选的历史行情数据源。
默认不会强制启用，只有在安装了 `baostock` 且登录成功时，
才会在 DataManager 中以名称 "baostock" 注册。

注意：
- Baostock 目前主要覆盖沪深 A 股，北交所支持有限；
- 这里只实现日线和股票列表接口，实时行情暂时返回空。
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import List

import pandas as pd
from loguru import logger

from ...config.settings import settings
from ...utils.decorators import retry_on_exception
from ..adapters import DataAdapter

try:  # 懒加载 baostock 依赖
    import baostock as bs

    BAOSTOCK_IMPORTED = True
except ImportError:  # pragma: no cover
    BAOSTOCK_IMPORTED = False
    logger.warning("baostock 未安装，如需使用 BaostockAdapter 请先安装 baostock")


def _symbol_to_bs_code(symbol: str) -> str:
    """将 6 位股票代码转换为 baostock code（如 sh.600000）。"""
    symbol = str(symbol).strip()
    if not symbol or len(symbol) != 6:
        return symbol

    if symbol.startswith(("60", "68")):
        return f"sh.{symbol}"
    # 默认深交所
    return f"sz.{symbol}"


class BaostockAdapter(DataAdapter):
    """基于 Baostock 的数据源适配器。"""

    def __init__(self) -> None:
        if not BAOSTOCK_IMPORTED:
            raise ImportError("baostock 库未安装，无法使用 BaostockAdapter")

        # Baostock 需要显示登录
        lg = bs.login()
        if lg.error_code != "0":  # pragma: no cover - 依赖外部服务
            raise RuntimeError(f"baostock 登录失败: {lg.error_msg}")

        self._logged_in = True

        # 简单节流控制，避免在多线程环境中过快请求 Baostock
        self._lock = threading.Lock()
        self._last_call_ts: float = 0.0
        self._min_interval: float = settings.baostock_min_interval

    def _throttle(self) -> None:
        """在多线程环境下对 Baostock 调用做简单节流。"""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_call_ts
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
                now = time.monotonic()
            self._last_call_ts = now

    # ------------------------------------------------------------------
    # 股票列表
    # ------------------------------------------------------------------
    @retry_on_exception(
        retries=settings.retry_count, delay=settings.retry_delay, backoff=2.0
    )
    def get_stock_list(self) -> pd.DataFrame:
        """获取在市 A 股列表，返回列：symbol, name, market。"""
        try:
            # Baostock 提供 stock_basic 接口，筛选在市 A 股
            self._throttle()
            rs = bs.query_stock_basic(
                code="", code_name="", org_id="", ipoDate="", outDate="", type="1"
            )
            if rs.error_code != "0":  # pragma: no cover
                logger.error(f"baostock 获取股票列表失败: {rs.error_msg}")
                return pd.DataFrame()

            data_list = []
            while rs.error_code == "0" and rs.next():  # pragma: no cover - 依赖外部服务
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)
            # 典型字段：code, code_name, ipoDate, outDate, type, status

            # 只保留在市股票
            if "status" in df.columns:
                df = df[df["status"] == "1"]

            # 拆分 code 为市场 + 代码
            def _split_code(code: str) -> tuple[str, str]:
                parts = str(code).split(".")
                if len(parts) == 2:
                    return parts[0], parts[1]
                return "", code

            markets = []
            symbols = []
            for code in df["code"]:
                m, sym = _split_code(code)
                symbols.append(sym)
                if m == "sh":
                    markets.append("上交所-主板")
                elif m == "sz":
                    markets.append("深交所-主板")
                else:
                    markets.append("其他")

            out = pd.DataFrame(
                {
                    "symbol": symbols,
                    "name": df["code_name"],
                    "market": markets,
                }
            )

            # 过滤 ST / 退市标记
            out = out[
                ~out["name"].str.contains("ST") & ~out["name"].str.contains("退")
            ].reset_index(drop=True)

            return out
        except Exception as e:  # pragma: no cover
            logger.error(f"BaostockAdapter.get_stock_list 失败: {e}")
            return pd.DataFrame()

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
        """获取日线数据，统一输出为内部标准格式。"""
        if not start_date:
            start_date = "2022-01-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        code = _symbol_to_bs_code(symbol)

        try:
            self._throttle()
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",  # 后复权
            )
            if rs.error_code != "0":  # pragma: no cover
                logger.error(f"baostock 获取 {code} 日线失败: {rs.error_msg}")
                return pd.DataFrame()

            data_list = []
            while rs.error_code == "0" and rs.next():  # pragma: no cover
                data_list.append(rs.get_row_data())

            if not data_list:
                return pd.DataFrame()

            df = pd.DataFrame(data_list, columns=rs.fields)

            # 数值转换
            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "turn",
                "pctChg",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])  # 保证日期有效
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            # 统一命名与补充字段
            df.rename(columns={"pctChg": "pct_chg"}, inplace=True)
            df["change"] = df["close"] - df["preclose"].fillna(df["close"].shift(1))
            df["amplitude"] = (
                (df["high"] - df["low"])
                / df["preclose"].fillna(df["close"].shift(1))
                * 100
            )

            result = df[
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
            ].dropna(subset=["open", "high", "low", "close"])

            return result.reset_index(drop=True)
        except Exception as e:  # pragma: no cover
            logger.error(f"BaostockAdapter.get_daily_data 失败 ({code}): {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 实时数据（占位）
    # ------------------------------------------------------------------
    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时数据。当前实现返回空 DataFrame。"""
        logger.warning("BaostockAdapter.get_realtime_data 暂未实现，返回空 DataFrame")
        return pd.DataFrame()

    def __del__(self) -> None:  # pragma: no cover
        if BAOSTOCK_IMPORTED and getattr(self, "_logged_in", False):
            try:
                bs.logout()
            except Exception as e:
                logger.warning(f"Baostock 登出失败: {e}")
