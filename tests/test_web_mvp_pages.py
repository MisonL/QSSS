"""Web MVP page smoke tests."""

import pandas as pd
import pytest
import sys
import threading
import time
import uuid
from pathlib import Path

pytest.importorskip("flask")
pytest.importorskip("celery")
pytest.importorskip("flask_migrate")
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


class _CountingBoardDataManager(_FakeDataManager):
    def __init__(self):
        self.board_calls = 0
        self.flow_calls = 0

    def get_boards(self, board_type="concept"):
        self.board_calls += 1
        return super().get_boards(board_type=board_type)

    def get_board_flows(self, board_type="concept"):
        self.flow_calls += 1
        return super().get_board_flows(board_type=board_type)


class _SlowCountingBoardDataManager(_CountingBoardDataManager):
    def get_boards(self, board_type="concept"):
        time.sleep(0.03)
        return super().get_boards(board_type=board_type)

    def get_board_flows(self, board_type="concept"):
        time.sleep(0.03)
        return super().get_board_flows(board_type=board_type)


class _FailingMarketDataManager(_FakeDataManager):
    def get_realtime_data(self, symbols):
        raise RuntimeError("secret-token-leak")


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

    market_html = client.get("/market").get_data(as_text=True)
    assert "指数行情" in market_html
    assert "默认自选池" in market_html

    boards_html = client.get("/boards").get_data(as_text=True)
    assert "板块强度" in boards_html
    assert "人工智能" in boards_html


def test_boards_page_uses_short_memory_cache(monkeypatch):
    """Boards page should not call slow data sources on every render."""
    fake_manager = _CountingBoardDataManager()
    monkeypatch.setattr(routes, "data_manager", fake_manager)
    routes._boards_cache.clear()
    app.config["TESTING"] = True
    client = app.test_client()

    assert client.get("/boards").status_code == 200
    assert client.get("/boards").status_code == 200

    assert fake_manager.board_calls == 1
    assert fake_manager.flow_calls == 1


def test_boards_page_serializes_concurrent_cache_misses(monkeypatch):
    fake_manager = _SlowCountingBoardDataManager()
    monkeypatch.setattr(routes, "data_manager", fake_manager)
    routes._boards_cache.clear()
    app.config["TESTING"] = True

    def request_boards():
        client = app.test_client()
        response = client.get("/boards")
        assert response.status_code == 200

    threads = [threading.Thread(target=request_boards) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert fake_manager.board_calls == 1
    assert fake_manager.flow_calls == 1


def test_page_data_source_error_is_masked(monkeypatch):
    """Raw data source exceptions should stay out of user-visible pages."""
    monkeypatch.setattr(routes, "data_manager", _FailingMarketDataManager())
    app.config["TESTING"] = True
    client = app.test_client()

    html = client.get("/market").get_data(as_text=True)

    assert "数据源请求失败，请查看服务端日志。" in html
    assert "secret-token-leak" not in html


def test_task_status_rejects_invalid_task_id(monkeypatch):
    called = False

    def forbidden_async_result(task_id):
        nonlocal called
        called = True
        raise AssertionError(f"unexpected AsyncResult call: {task_id}")

    monkeypatch.setattr(routes.celery, "AsyncResult", forbidden_async_result)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get("/api/task/" + "x" * 200)

    assert response.status_code == 400
    assert response.get_json() == {"error": "任务 ID 格式无效"}
    assert called is False


def test_task_status_accepts_uuid_task_id(monkeypatch):
    task_id = str(uuid.uuid4())
    seen_ids = []

    class _PendingTask:
        state = "PENDING"

    def fake_async_result(received_task_id):
        seen_ids.append(received_task_id)
        return _PendingTask()

    monkeypatch.setattr(routes.celery, "AsyncResult", fake_async_result)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.get(f"/api/task/{task_id}")

    assert response.status_code == 200
    assert response.get_json()["state"] == "PENDING"
    assert seen_ids == [task_id]
