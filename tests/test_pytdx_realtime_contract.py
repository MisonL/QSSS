"""Pytdx realtime quote field contract tests."""

from qsss.data.adapters import PytdxAdapter


class _FakePytdxApi:
    def __init__(self):
        self.calls = []

    def get_security_quotes(self, securities):
        self.calls.append(securities)
        return [
            {
                "price": 12.34,
                "last_close": 12.0,
                "open": 12.1,
                "high": 12.8,
                "low": 11.9,
                "vol": 123456,
                "amount": 3456789.0,
                "servertime": "2026-05-12 14:30:00",
            }
        ]

    def disconnect(self):
        return None


def test_pytdx_realtime_maps_native_quote_fields():
    """pytdx native quote fields should map to QSSS realtime contract."""
    adapter = PytdxAdapter()
    fake_api = _FakePytdxApi()
    adapter.api = fake_api
    adapter._connected = True

    df = adapter.get_realtime_data(["000001"])

    assert fake_api.calls == [[(0, "000001")]]
    assert len(df) == 1
    row = df.iloc[0]

    assert row["symbol"] == "000001"
    assert row["name"] == ""
    assert row["price"] == 12.34
    assert row["last_close"] == 12.0
    assert row["open"] == 12.1
    assert row["high"] == 12.8
    assert row["low"] == 11.9
    assert row["volume"] == 123456
    assert row["amount"] == 3456789.0
    assert row["quote_time"] == "2026-05-12 14:30:00"
    assert row["source"] == "pytdx"
    assert isinstance(row["fetched_at"], str)
