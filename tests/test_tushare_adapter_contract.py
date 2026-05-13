"""Tushare adapter contract tests that do not require a live token."""

import pandas as pd

from qsss.data.sources import tushare_adapter as module
from qsss.data.sources.tushare_adapter import TushareAdapter


class _FakePro:
    def stock_basic(self, exchange, list_status, fields):
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "平安银行",
                    "market": "主板",
                    "list_status": "L",
                },
                {
                    "ts_code": "688001.SH",
                    "symbol": "688001",
                    "name": "测试科技",
                    "market": "科创板",
                    "list_status": "L",
                },
            ]
        )

    def daily(self, ts_code, start_date, end_date):
        return pd.DataFrame(
            [
                {
                    "trade_date": "20230103",
                    "open": 10.0,
                    "high": 11.0,
                    "low": 9.5,
                    "close": 10.5,
                    "vol": 100000,
                    "amount": 1000000,
                    "pct_chg": 5.0,
                },
                {
                    "trade_date": "20230104",
                    "open": 10.5,
                    "high": 12.0,
                    "low": 10.0,
                    "close": 11.5,
                    "vol": 110000,
                    "amount": 1200000,
                    "pct_chg": 9.52,
                },
            ]
        )

    def daily_basic(self, ts_code, start_date, end_date, fields):
        return pd.DataFrame(
            [
                {"trade_date": "20230103", "turnover_rate": 1.2},
                {"trade_date": "20230104", "turnover_rate": 1.4},
            ]
        )


class _FakeTushare:
    def __init__(self):
        self.token = None

    def set_token(self, token):
        self.token = token

    def pro_api(self, token):
        self.token = token
        return _FakePro()

    def get_realtime_quotes(self, codes):
        return pd.DataFrame(
            [
                {
                    "code": "000001",
                    "name": "平安银行",
                    "price": "11.0",
                    "pre_close": "10.5",
                    "open": "10.6",
                    "high": "11.2",
                    "low": "10.4",
                    "volume": "100000",
                    "amount": "1100000",
                    "time": "14:30:00",
                }
            ]
        )


def _adapter(monkeypatch):
    fake_ts = _FakeTushare()
    monkeypatch.setattr(module, "ts", fake_ts, raising=False)
    monkeypatch.setattr(module, "TUSHARE_IMPORTED", True)
    return TushareAdapter(token="test-token")


def test_tushare_stock_list_field_mapping(monkeypatch):
    df = _adapter(monkeypatch).get_stock_list()

    assert df.to_dict("records") == [
        {"symbol": "000001", "name": "平安银行", "market": "深交所-主板"},
        {"symbol": "688001", "name": "测试科技", "market": "上交所-科创板"},
    ]


def test_tushare_daily_field_mapping(monkeypatch):
    df = _adapter(monkeypatch).get_daily_data("000001", "20230101", "20230131")

    assert list(df.columns) == [
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
    assert df["date"].tolist() == ["2023-01-03", "2023-01-04"]
    assert df["turn"].tolist() == [1.2, 1.4]


def test_tushare_realtime_field_mapping(monkeypatch):
    df = _adapter(monkeypatch).get_realtime_data(["000001"])

    row = df.iloc[0].to_dict()
    assert row["symbol"] == "000001"
    assert row["name"] == "平安银行"
    assert row["price"] == 11.0
    assert row["last_close"] == 10.5
    assert row["time"] == "14:30:00"
