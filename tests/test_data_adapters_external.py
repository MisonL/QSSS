"""Integration tests for external data adapters (pytdx / Tushare / Baostock).

These tests hit real upstream data providers and therefore are:
- marked with pytest.mark.external
- designed to use small samples
- skipped automatically when the corresponding provider or credentials
  are not available.

They focus on validating the *field contract* and basic invariants
rather than asserting specific numeric values.
"""

import os

import pytest

from qsss.data.adapters import PYTDX_AVAILABLE, PytdxAdapter
from qsss.data.sources.akshare_adapter import AKSHARE_AVAILABLE, AkshareAdapter
from qsss.data.sources.baostock_adapter import BAOSTOCK_IMPORTED, BaostockAdapter
from qsss.data.sources.tushare_adapter import TUSHARE_IMPORTED, TushareAdapter

REQUIRED_DAILY_COLUMNS = [
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

REQUIRED_STOCK_LIST_COLUMNS = ["symbol", "name", "market"]
REQUIRED_BOARD_COLUMNS = [
    "board_code",
    "board_name",
    "board_type",
    "pct_chg",
    "amount",
    "source",
    "fetched_at",
]
REQUIRED_BOARD_FLOW_COLUMNS = [
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

SAMPLE_SYMBOL = "000001"  # 深交所常见样本代码
SAMPLE_START = "20230101"
SAMPLE_END = "20230401"


# ---------------------------------------------------------------------------
# AkshareAdapter
# ---------------------------------------------------------------------------


@pytest.mark.external
def test_akshare_concept_board_list_schema_small_sample():
    """AKShare 概念板块列表字段与内部统一格式保持一致。"""
    if not AKSHARE_AVAILABLE:
        pytest.skip("akshare 未安装，跳过 AkshareAdapter 集成测试")

    try:
        adapter = AkshareAdapter()
        df = adapter.get_boards(board_type="concept")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AKShare 概念板块列表获取失败（可能是上游或网络问题）：{exc}")

    if df is None or df.empty:
        pytest.skip("AKShare 返回空概念板块列表，可能是上游服务不可用")

    for col in REQUIRED_BOARD_COLUMNS:
        assert col in df.columns
    assert set(df["source"]) == {"akshare"}


@pytest.mark.external
def test_akshare_concept_board_flow_schema_small_sample():
    """AKShare 概念板块资金流字段与内部统一格式保持一致。"""
    if not AKSHARE_AVAILABLE:
        pytest.skip("akshare 未安装，跳过 AkshareAdapter 集成测试")

    try:
        adapter = AkshareAdapter()
        df = adapter.get_board_flows(board_type="concept")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"AKShare 概念板块资金流获取失败（可能是上游或网络问题）：{exc}")

    if df is None or df.empty:
        pytest.skip("AKShare 返回空概念板块资金流，可能是上游服务不可用")

    for col in REQUIRED_BOARD_FLOW_COLUMNS:
        assert col in df.columns
    assert set(df["source"]) == {"akshare"}


# ---------------------------------------------------------------------------
# PytdxAdapter
# ---------------------------------------------------------------------------


@pytest.mark.external
def test_pytdx_stock_list_contract():
    """Pytdx 股票列表字段与过滤规则保持契约一致。"""
    if not PYTDX_AVAILABLE:
        pytest.skip("pytdx 未安装，跳过 PytdxAdapter 集成测试")

    adapter = PytdxAdapter()
    df = adapter.get_stock_list()

    if df is None or df.empty:
        pytest.skip("pytdx 返回空股票列表，可能是上游服务不可用")

    # 字段契约
    for col in REQUIRED_STOCK_LIST_COLUMNS:
        assert col in df.columns

    # ST / 退市股票应已被过滤
    assert not df["name"].astype(str).str.contains("ST").any()
    assert not df["name"].astype(str).str.contains("退").any()


@pytest.mark.external
def test_pytdx_daily_schema_small_sample():
    """Pytdx 日线数据字段与内部统一格式保持一致。"""
    if not PYTDX_AVAILABLE:
        pytest.skip("pytdx 未安装，跳过 PytdxAdapter 集成测试")

    adapter = PytdxAdapter()
    df = adapter.get_daily_data(SAMPLE_SYMBOL, SAMPLE_START, SAMPLE_END)

    if df is None or df.empty:
        pytest.skip("pytdx 返回空日线数据，可能是上游服务或网络问题")

    for col in REQUIRED_DAILY_COLUMNS:
        assert col in df.columns


# ---------------------------------------------------------------------------
# TushareAdapter
# ---------------------------------------------------------------------------


@pytest.mark.external
def test_tushare_stock_list_contract_when_configured():
    """Tushare 股票列表字段与过滤规则保持契约一致（在配置 token 时）。"""
    if not TUSHARE_IMPORTED:
        pytest.skip("tushare 未安装，跳过 TushareAdapter 集成测试")

    token = os.getenv("QSSS_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if not token:
        pytest.skip("未配置 Tushare Token，跳过 TushareAdapter 集成测试")

    try:
        adapter = TushareAdapter()
    except Exception as exc:  # noqa: BLE001 - 这里只是将异常转化为跳过
        pytest.skip(f"TushareAdapter 初始化失败（可能是凭据或网络问题）：{exc}")

    df = adapter.get_stock_list()
    if df is None or df.empty:
        pytest.skip("Tushare 返回空股票列表，可能是上游服务不可用")

    for col in REQUIRED_STOCK_LIST_COLUMNS:
        assert col in df.columns

    # ST / 退市股票应已被过滤
    assert not df["name"].astype(str).str.contains("ST").any()
    assert not df["name"].astype(str).str.contains("退").any()


@pytest.mark.external
def test_tushare_daily_schema_small_sample_when_configured():
    """Tushare 日线数据字段与内部统一格式保持一致（在配置 token 时）。"""
    if not TUSHARE_IMPORTED:
        pytest.skip("tushare 未安装，跳过 TushareAdapter 集成测试")

    token = os.getenv("QSSS_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if not token:
        pytest.skip("未配置 Tushare Token，跳过 TushareAdapter 集成测试")

    try:
        adapter = TushareAdapter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"TushareAdapter 初始化失败（可能是凭据或网络问题）：{exc}")

    df = adapter.get_daily_data(SAMPLE_SYMBOL, SAMPLE_START, SAMPLE_END)
    if df is None or df.empty:
        pytest.skip("Tushare 返回空日线数据，可能是上游服务不可用")

    for col in REQUIRED_DAILY_COLUMNS:
        assert col in df.columns


# ---------------------------------------------------------------------------
# BaostockAdapter
# ---------------------------------------------------------------------------


@pytest.mark.external
def test_tushare_realtime_schema_when_configured():
    """Tushare 实时行情字段与 PytdxAdapter 输出保持基本一致（在配置 token 时）。"""
    if not TUSHARE_IMPORTED:
        pytest.skip("tushare 未安装，跳过 TushareAdapter 实时行情集成测试")

    token = os.getenv("QSSS_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN")
    if not token:
        pytest.skip("未配置 Tushare Token，跳过 TushareAdapter 实时行情集成测试")

    try:
        adapter = TushareAdapter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"TushareAdapter 初始化失败（可能是凭据或网络问题）：{exc}")

    df = adapter.get_realtime_data([SAMPLE_SYMBOL])
    if df is None or df.empty:
        pytest.skip("Tushare 实时行情返回空，可能是上游服务不可用")

    for col in ["symbol", "name", "price", "open", "high", "low"]:
        assert col in df.columns

    # ---------------------------------------------------------------------------
    # BaostockAdapter
    # ---------------------------------------------------------------------------


@pytest.mark.external
def test_baostock_stock_list_contract_when_available():
    """Baostock 股票列表字段与过滤规则保持契约一致（在安装并登录成功时）。"""
    if not BAOSTOCK_IMPORTED:
        pytest.skip("baostock 未安装，跳过 BaostockAdapter 集成测试")

    try:
        adapter = BaostockAdapter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BaostockAdapter 初始化失败（可能是登录或网络问题）：{exc}")

    df = adapter.get_stock_list()
    if df is None or df.empty:
        pytest.skip("Baostock 返回空股票列表，可能是上游服务不可用")

    for col in REQUIRED_STOCK_LIST_COLUMNS:
        assert col in df.columns

    assert not df["name"].astype(str).str.contains("ST").any()
    assert not df["name"].astype(str).str.contains("退").any()


@pytest.mark.external
def test_baostock_daily_schema_small_sample_when_available():
    """Baostock 日线数据字段与内部统一格式保持一致（在安装并登录成功时）。"""
    if not BAOSTOCK_IMPORTED:
        pytest.skip("baostock 未安装，跳过 BaostockAdapter 集成测试")

    try:
        adapter = BaostockAdapter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BaostockAdapter 初始化失败（可能是登录或网络问题）：{exc}")

    df = adapter.get_daily_data(SAMPLE_SYMBOL, SAMPLE_START, SAMPLE_END)
    if df is None or df.empty:
        pytest.skip("Baostock 返回空日线数据，可能是上游服务不可用")

    for col in REQUIRED_DAILY_COLUMNS:
        assert col in df.columns


@pytest.mark.external
def test_baostock_realtime_schema_when_available():
    """Baostock 实时行情在可用环境下返回非空并包含价格类字段。"""
    if not BAOSTOCK_IMPORTED:
        pytest.skip("baostock 未安装，跳过 BaostockAdapter 实时行情集成测试")

    try:
        adapter = BaostockAdapter()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"BaostockAdapter 初始化失败（可能是登录或网络问题）：{exc}")

    try:
        df = adapter.get_realtime_data([SAMPLE_SYMBOL])
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    if df is None or df.empty:
        pytest.skip("Baostock 实时行情返回空，可能是上游服务不可用")

    # 不强制字段完全对齐，只检查有 code/price 等基本字段即可
    assert any(col in df.columns for col in ["symbol", "code"])
    assert any(col in df.columns for col in ["price", "close"])
