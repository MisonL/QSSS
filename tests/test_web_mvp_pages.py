"""Web MVP page smoke tests."""

import pandas as pd
import pytest
import sys
from pathlib import Path

pytest.importorskip("flask")
pytest.importorskip("flask_sqlalchemy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.app import app
import web.routes as routes


class _FakeDataManager:
    def get_stock_list(self):
        return pd.DataFrame(
            [
                {"symbol": "300033", "name": "同花顺", "market": "深交所-创业板"},
                {"symbol": "000977", "name": "浪潮信息", "market": "深交所-主板"},
            ]
        )

    def get_realtime_data(self, symbols):
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "name": f"名称{symbol}",
                    "price": 10.5,
                    "pct_chg": 1.2,
                    "source": "pytdx",
                }
                for symbol in symbols
            ]
        )

    def get_boards(self, board_type="concept"):
        return pd.DataFrame(
            [
                {
                    "board_code": "BK0800",
                    "board_name": "人工智能",
                    "board_type": board_type,
                    "pct_chg": 2.5,
                    "amount": 123000000,
                    "source": "akshare",
                }
            ]
        )

    def get_board_flows(self, board_type="concept"):
        return pd.DataFrame(
            [
                {
                    "board_code": "BK0800",
                    "board_name": "人工智能",
                    "board_type": board_type,
                    "net_inflow": 23000000,
                    "main_net_inflow": 23000000,
                    "source": "akshare",
                }
            ]
        )


def test_web_mvp_pages_render_real_data_surfaces(monkeypatch):
    """MVP pages should render market, board and AI watchlist data surfaces."""
    monkeypatch.setattr(routes, "data_manager", _FakeDataManager())
    app.config["TESTING"] = True
    client = app.test_client()

    pages = [
        ("/market", "行情总览"),
        ("/boards", "板块轮动"),
        ("/watchlists/ai-tech", "AI 科技观察池"),
    ]

    for path, expected_text in pages:
        response = client.get(path)
        assert response.status_code == 200
        assert expected_text in response.get_data(as_text=True)
