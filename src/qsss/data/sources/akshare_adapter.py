"""akshare数据源适配器"""

from datetime import datetime
from hashlib import sha1
from typing import List, Optional

import pandas as pd
from loguru import logger

try:
    import akshare as ak

    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("akshare未安装")


class AkshareAdapter:
    """akshare数据源适配器"""

    def __init__(self) -> None:
        if not AKSHARE_AVAILABLE:
            raise ImportError("akshare库未安装")

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        try:
            stock_info = ak.stock_zh_a_spot_em()
            stock_info = stock_info[["代码", "名称"]].copy()
            stock_info.columns = ["symbol", "name"]

            # 过滤ST和退市股票
            stock_info = stock_info[
                ~stock_info["name"].str.contains("退市|退")
                & ~stock_info["name"].str.contains("ST")
            ]

            # 添加市场信息
            def get_market_info(symbol: str) -> str:
                if symbol.startswith("60"):
                    return "上交所-主板"
                elif symbol.startswith("688"):
                    return "上交所-科创板"
                elif symbol.startswith("000"):
                    return "深交所-主板"
                elif symbol.startswith("002"):
                    return "深交所-中小板"
                elif symbol.startswith("300"):
                    return "深交所-创业板"
                elif symbol.startswith("301"):
                    return "深交所-创业板"
                else:
                    return "其他"

            stock_info["market"] = stock_info["symbol"].apply(get_market_info)

            return stock_info

        except Exception as e:
            logger.error(f"akshare获取股票列表失败: {e}")
            return pd.DataFrame()

    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取日线数据"""
        try:
            if not start_date:
                start_date = "20220101"
            if not end_date:
                end_date = datetime.now().strftime("%Y%m%d")

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            if df.empty:
                return df

            # 重命名列
            df = df[
                [
                    "日期",
                    "开盘",
                    "收盘",
                    "最高",
                    "最低",
                    "成交量",
                    "成交额",
                    "振幅",
                    "涨跌幅",
                    "涨跌额",
                    "换手率",
                ]
            ]
            df.columns = [
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

            # 数据类型转换
            numeric_columns = [
                "open",
                "close",
                "high",
                "low",
                "volume",
                "amount",
                "turn",
                "pct_chg",
            ]
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna()
            return df

        except Exception as e:
            logger.error(f"akshare获取{symbol}数据失败: {e}")
            return pd.DataFrame()

    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时数据"""
        # akshare实时数据功能有限，返回空DataFrame
        return pd.DataFrame()

    def get_boards(self, board_type: str = "concept") -> pd.DataFrame:
        """获取 AKShare 板块列表并规范为 QSSS 字段。"""
        board_type = _normalize_board_type(board_type)
        errors = []
        for source_name, fetcher in _get_board_name_fetchers(board_type):
            try:
                raw_df = fetcher()
                if raw_df is None or raw_df.empty:
                    raise ValueError(f"AKShare {source_name} 板块列表为空")
                return _normalize_board_list(raw_df, board_type)
            except Exception as e:
                errors.append(f"{source_name}: {e}")
                logger.warning(f"AKShare {source_name} 板块列表获取失败: {e}")
        error_detail = "; ".join(errors)
        message = f"AKShare 获取 {board_type} 板块列表失败: {error_detail}"
        logger.error(message)
        raise RuntimeError(message)

    def get_board_flows(self, board_type: str = "concept") -> pd.DataFrame:
        """获取 AKShare 板块资金流并规范为 QSSS 字段。"""
        board_type = _normalize_board_type(board_type)
        fetcher = _get_board_flow_fetcher(board_type)
        try:
            raw_df = fetcher(symbol="即时")
            if raw_df is None or raw_df.empty:
                raise ValueError(f"AKShare {board_type} 板块资金流为空")
            return _normalize_board_flows(raw_df, board_type)
        except Exception as e:
            message = f"AKShare 获取 {board_type} 板块资金流失败: {e}"
            logger.error(message)
            raise RuntimeError(message) from e


def _normalize_board_type(board_type: str) -> str:
    normalized = str(board_type).strip().lower()
    if normalized not in {"concept", "industry"}:
        raise ValueError(f"不支持的板块类型: {board_type}")
    return normalized


def _get_board_name_fetchers(board_type: str):
    if board_type == "concept":
        candidates = [
            ("concept_em", "stock_board_concept_name_em"),
            ("concept_ths", "stock_board_concept_name_ths"),
        ]
    else:
        candidates = [
            ("industry_em", "stock_board_industry_name_em"),
            ("industry_ths", "stock_board_industry_name_ths"),
        ]
    return [
        (source_name, getattr(ak, function_name))
        for source_name, function_name in candidates
        if hasattr(ak, function_name)
    ]


def _get_board_flow_fetcher(board_type: str):
    if board_type == "concept":
        return ak.stock_fund_flow_concept
    return ak.stock_fund_flow_industry


def _first_existing_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"AKShare 返回缺少字段 {label}: {list(df.columns)}")


def _optional_numeric(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def _stable_board_code(board_name: str) -> str:
    digest = sha1(board_name.encode("utf-8")).hexdigest()[:10].upper()
    return f"AK{digest}"


def _normalize_board_list(raw_df: pd.DataFrame, board_type: str) -> pd.DataFrame:
    name_col = _first_existing_column(
        raw_df, ["板块名称", "行业", "名称", "name"], "板块名称"
    )
    code_col = next(
        (col for col in ["板块代码", "代码", "code"] if col in raw_df.columns), None
    )
    result = pd.DataFrame(index=raw_df.index)
    result["board_name"] = raw_df[name_col].astype(str).str.strip()
    result["board_code"] = (
        raw_df[code_col].astype(str).str.strip()
        if code_col
        else result["board_name"].map(_stable_board_code)
    )
    result["board_type"] = board_type
    result["pct_chg"] = _optional_numeric(raw_df, ["涨跌幅", "行业-涨跌幅"])
    result["amount"] = _optional_numeric(raw_df, ["成交额", "流入资金"])
    result["source"] = "akshare"
    result["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return result[
        [
            "board_code",
            "board_name",
            "board_type",
            "pct_chg",
            "amount",
            "source",
            "fetched_at",
        ]
    ]


def _normalize_board_flows(raw_df: pd.DataFrame, board_type: str) -> pd.DataFrame:
    name_col = _first_existing_column(raw_df, ["板块名称", "行业", "名称"], "板块名称")
    code_col = next(
        (col for col in ["板块代码", "代码"] if col in raw_df.columns), None
    )
    result = pd.DataFrame(index=raw_df.index)
    result["board_name"] = raw_df[name_col].astype(str).str.strip()
    result["board_code"] = (
        raw_df[code_col].astype(str).str.strip()
        if code_col
        else result["board_name"].map(_stable_board_code)
    )
    result["board_type"] = board_type
    result["pct_chg"] = _optional_numeric(raw_df, ["行业-涨跌幅", "涨跌幅"])
    result["amount"] = _optional_numeric(raw_df, ["成交额", "流入资金"])
    result["net_inflow"] = _optional_numeric(raw_df, ["净额", "净流入"])
    result["main_net_inflow"] = _optional_numeric(raw_df, ["主力净流入", "净额"])
    result["source"] = "akshare"
    result["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return result[
        [
            "board_code",
            "board_name",
            "board_type",
            "pct_chg",
            "amount",
            "net_inflow",
            "main_net_inflow",
            "source",
            "fetched_at",
        ]
    ]
