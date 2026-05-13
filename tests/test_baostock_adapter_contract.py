"""Baostock adapter contract tests that do not hit the network."""

from qsss.data.sources import baostock_adapter as module
from qsss.data.sources.baostock_adapter import BaostockAdapter


class _FakeLoginResult:
    error_code = "0"
    error_msg = "success"


class _FakeResultSet:
    error_code = "0"
    error_msg = "success"

    def __init__(self, fields, rows):
        self.fields = fields
        self._rows = rows
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class _FakeBaostock:
    def __init__(self):
        self.history_args = None

    def login(self):
        return _FakeLoginResult()

    def query_stock_basic(self):
        return _FakeResultSet(
            ["code", "code_name", "ipoDate", "outDate", "type", "status"],
            [
                ["sh.600000", "浦发银行", "1999-11-10", "", "1", "1"],
                ["sh.000001", "上证综合指数", "1991-07-15", "", "2", "1"],
                ["sz.000001", "平安银行", "1991-04-03", "", "1", "1"],
            ],
        )

    def query_history_k_data_plus(
        self, code, fields, start_date, end_date, frequency, adjustflag
    ):
        self.history_args = {
            "code": code,
            "start_date": start_date,
            "end_date": end_date,
        }
        return _FakeResultSet(
            fields.split(","),
            [
                [
                    "2023-01-03",
                    code,
                    "10.0",
                    "11.0",
                    "9.5",
                    "10.5",
                    "9.8",
                    "100000",
                    "1000000",
                    "1.1",
                    "7.14",
                ],
            ],
        )

    def logout(self):
        return None


def test_baostock_stock_list_filters_a_share_rows(monkeypatch):
    fake_bs = _FakeBaostock()
    monkeypatch.setattr(module, "bs", fake_bs)
    monkeypatch.setattr(module, "BAOSTOCK_IMPORTED", True)

    df = BaostockAdapter().get_stock_list()

    assert df["symbol"].tolist() == ["600000", "000001"]
    assert df["name"].tolist() == ["浦发银行", "平安银行"]


def test_baostock_daily_data_normalizes_compact_dates(monkeypatch):
    fake_bs = _FakeBaostock()
    monkeypatch.setattr(module, "bs", fake_bs)
    monkeypatch.setattr(module, "BAOSTOCK_IMPORTED", True)

    df = BaostockAdapter().get_daily_data("000001", "20230101", "20230401")

    assert fake_bs.history_args == {
        "code": "sz.000001",
        "start_date": "2023-01-01",
        "end_date": "2023-04-01",
    }
    assert not df.empty
    assert "pct_chg" in df.columns


def test_baostock_realtime_is_explicitly_unsupported(monkeypatch):
    fake_bs = _FakeBaostock()
    monkeypatch.setattr(module, "bs", fake_bs)
    monkeypatch.setattr(module, "BAOSTOCK_IMPORTED", True)

    adapter = BaostockAdapter()

    try:
        adapter.get_realtime_data(["000001"])
    except NotImplementedError as exc:
        assert "不支持实时行情" in str(exc)
    else:
        raise AssertionError("Baostock realtime should be explicitly unsupported")
