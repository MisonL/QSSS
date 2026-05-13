"""CLI analyze command contract tests."""

import pandas as pd
from click.testing import CliRunner

from qsss import cli as cli_module


class _FakeStrategy:
    calls = []
    ma_calls = []

    def run_analysis(self, start_date="20220101", limit=None):
        self.__class__.calls.append({"start_date": start_date, "limit": limit})
        return pd.DataFrame(
            [
                {
                    "name": "测试股票",
                    "symbol": "000001",
                    "market": "深交所-主板",
                    "prediction": 0.8,
                    "momentum_score": 0.2,
                    "rsi": 50,
                    "close": 10.0,
                    "explosion_score": 2.0,
                    "macd_status": "金叉",
                }
            ]
        )

    def calculate_ma15(self, symbol, start_date="20220101"):
        self.__class__.ma_calls.append({"symbol": symbol, "start_date": start_date})
        return 9.5


def test_cli_analyze_passes_start_date_and_limit(monkeypatch):
    """CLI analyze should pass start-date and limit into strategy flow."""
    _FakeStrategy.calls = []
    _FakeStrategy.ma_calls = []
    monkeypatch.setattr(cli_module, "QuantStrategy", _FakeStrategy)

    result = CliRunner().invoke(
        cli_module.cli,
        ["analyze", "--start-date", "20240101", "--limit", "1"],
    )

    assert result.exit_code == 0
    assert _FakeStrategy.calls == [{"start_date": "20240101", "limit": 1}]
    assert _FakeStrategy.ma_calls == [{"symbol": "000001", "start_date": "20240101"}]
