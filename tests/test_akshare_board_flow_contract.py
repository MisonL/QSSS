"""AKShare board and fund-flow contract tests."""

import pandas as pd

from qsss.data.manager import DataManager
from qsss.data.sources import akshare_adapter
from qsss.data.sources.akshare_adapter import AkshareAdapter


class _FakeAkshare:
    def stock_fund_flow_concept(self, symbol="即时"):
        assert symbol == "即时"
        return pd.DataFrame(
            [
                {
                    "行业": "人工智能",
                    "行业-涨跌幅": 2.5,
                    "流入资金": 150000000.0,
                    "净额": 23000000.0,
                }
            ]
        )

    def stock_board_concept_name_em(self):
        return pd.DataFrame(
            [
                {
                    "板块代码": "BK0800",
                    "板块名称": "人工智能",
                    "涨跌幅": 2.5,
                    "成交额": 1230000000.0,
                }
            ]
        )


class _FakeAkshareWithBoardListFallback(_FakeAkshare):
    def stock_board_concept_name_em(self):
        raise ConnectionError("eastmoney closed connection")

    def stock_board_concept_name_ths(self):
        return pd.DataFrame([{"name": "AI PC", "code": "309121"}])


def test_akshare_board_flows_normalize_concept_fund_flow(monkeypatch):
    """AKShare concept fund-flow columns should map to QSSS board contract."""
    monkeypatch.setattr(akshare_adapter, "AKSHARE_AVAILABLE", True)
    monkeypatch.setattr(akshare_adapter, "ak", _FakeAkshare())

    adapter = AkshareAdapter()
    df = adapter.get_board_flows(board_type="concept")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["board_name"] == "人工智能"
    assert row["board_type"] == "concept"
    assert row["pct_chg"] == 2.5
    assert row["amount"] == 150000000.0
    assert row["net_inflow"] == 23000000.0
    assert row["main_net_inflow"] == 23000000.0
    assert row["source"] == "akshare"
    assert isinstance(row["fetched_at"], str)


def test_akshare_board_list_normalizes_concept_boards(monkeypatch):
    """AKShare concept board list should expose stable board fields."""
    monkeypatch.setattr(akshare_adapter, "AKSHARE_AVAILABLE", True)
    monkeypatch.setattr(akshare_adapter, "ak", _FakeAkshare())

    adapter = AkshareAdapter()
    df = adapter.get_boards(board_type="concept")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["board_code"] == "BK0800"
    assert row["board_name"] == "人工智能"
    assert row["board_type"] == "concept"
    assert row["pct_chg"] == 2.5
    assert row["amount"] == 1230000000.0
    assert row["source"] == "akshare"


def test_akshare_board_list_uses_ths_when_em_unavailable(monkeypatch):
    """AKShare board list should use explicit THS fallback when EM fails."""
    monkeypatch.setattr(akshare_adapter, "AKSHARE_AVAILABLE", True)
    monkeypatch.setattr(akshare_adapter, "ak", _FakeAkshareWithBoardListFallback())

    adapter = AkshareAdapter()
    df = adapter.get_boards(board_type="concept")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["board_code"] == "309121"
    assert row["board_name"] == "AI PC"
    assert row["board_type"] == "concept"
    assert row["source"] == "akshare"


def test_data_manager_exposes_akshare_board_flows(monkeypatch):
    """DataManager should route board flows to AKShare without realtime takeover."""
    monkeypatch.setattr(akshare_adapter, "AKSHARE_AVAILABLE", True)
    monkeypatch.setattr(akshare_adapter, "ak", _FakeAkshare())

    manager = DataManager()
    df = manager.get_board_flows(board_type="concept")

    assert "akshare" in manager.get_available_sources()
    assert len(df) == 1
    assert set(df["source"]) == {"akshare"}


def test_data_manager_exposes_akshare_boards(monkeypatch):
    """DataManager should route board lists to AKShare board capability."""
    monkeypatch.setattr(akshare_adapter, "AKSHARE_AVAILABLE", True)
    monkeypatch.setattr(akshare_adapter, "ak", _FakeAkshare())

    manager = DataManager()
    df = manager.get_boards(board_type="concept")

    assert "akshare" in manager.get_available_sources()
    assert len(df) == 1
    assert set(df["board_name"]) == {"人工智能"}
    assert set(df["source"]) == {"akshare"}
