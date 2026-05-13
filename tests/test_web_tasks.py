"""Web task helper contract tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("celery")
pytest.importorskip("flask")
pytest.importorskip("flask_migrate")
pytest.importorskip("flask_sqlalchemy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web.routes as routes  # noqa: E402
import web.tasks as web_tasks  # noqa: E402
from web.app import app, db  # noqa: E402
from web.models import Backtest, Stock, Strategy  # noqa: E402
from web.tasks import _resolve_stock_info, run_backtest_task  # noqa: E402


class _StockListDataManager:
    def get_stock_list(self):
        return pd.DataFrame(
            [
                {"symbol": "000001", "name": "平安银行", "market": "SZ"},
                {"symbol": "600519", "name": "贵州茅台", "market": "SH"},
            ]
        )


class _FailingStockListDataManager:
    def get_stock_list(self):
        raise RuntimeError("metadata unavailable")


class _BacktestDataManager(_StockListDataManager):
    def get_daily_data(self, symbol, start_date=None, end_date=None, source=None):
        return pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "close": [10.0, 11.0, 12.0],
                "open": [10.0, 10.5, 11.5],
                "high": [10.5, 11.5, 12.5],
                "low": [9.5, 10.5, 11.5],
                "volume": [100000, 110000, 120000],
            }
        )


class _EmptyBacktestDataManager(_StockListDataManager):
    def get_daily_data(self, symbol, start_date=None, end_date=None, source=None):
        return pd.DataFrame()


class _UnexpectedDataManager:
    def __init__(self):
        raise AssertionError("unsupported strategy should not initialize data manager")


def test_resolve_stock_info_uses_data_manager_metadata():
    stock_info = _resolve_stock_info(_StockListDataManager(), "600519")

    assert stock_info == {"symbol": "600519", "name": "贵州茅台", "market": "SH"}


def test_resolve_stock_info_falls_back_when_metadata_unavailable():
    stock_info = _resolve_stock_info(_FailingStockListDataManager(), "600519")

    assert stock_info == {"symbol": "600519", "name": "600519", "market": "Unknown"}


def test_backtest_api_runs_minimal_backtest(monkeypatch):
    monkeypatch.setattr(routes, "data_manager", _BacktestDataManager())
    app.config["TESTING"] = True
    assert app.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"
    with app.app_context():
        db.create_all()
        try:
            client = app.test_client()
            response = client.post(
                "/api/backtest",
                json={
                    "stock_code": "399998",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "initial_capital": 100000,
                },
            )

            assert response.status_code == 200
            task_id = response.get_json()["task_id"]
            status = client.get(f"/api/task/{task_id}")
            result = status.get_json()

            assert result["state"] == "SUCCESS"
            assert result["result"]["total_trades"] == 1
            assert result["result"]["total_return"] > 0
        finally:
            db.session.remove()
            db.drop_all()


def test_run_backtest_task_rejects_unsupported_strategy_before_data_fetch(
    monkeypatch,
):
    monkeypatch.setattr(web_tasks, "DataManager", _UnexpectedDataManager)
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        try:
            stock = Stock(code="399998", name="测试股票", market="SZ")
            strategy = Strategy(
                name="技术分析策略",
                description="测试用策略",
                type="technical",
                parameters={},
                is_active=True,
            )
            db.session.add_all([stock, strategy])
            db.session.commit()
            backtest = Backtest(
                stock_id=stock.id,
                strategy_id=strategy.id,
                start_date=pd.Timestamp("2024-01-01").date(),
                end_date=pd.Timestamp("2024-01-03").date(),
                initial_capital=100000,
                status="pending",
            )
            db.session.add(backtest)
            db.session.commit()

            result = run_backtest_task.run(backtest.id)

            assert result == {
                "status": "failed",
                "error": "回测任务失败，请查看服务端日志。",
                "backtest_id": backtest.id,
            }
            assert db.session.get(Backtest, backtest.id).status == "failed"
        finally:
            db.session.remove()
            db.drop_all()


def test_run_backtest_task_rejects_missing_stock_before_data_fetch(
    monkeypatch,
):
    monkeypatch.setattr(web_tasks, "DataManager", _UnexpectedDataManager)
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        try:
            strategy = Strategy(
                name="买入持有基线",
                description="测试用策略",
                type="buy_hold",
                parameters={},
                is_active=True,
            )
            db.session.add(strategy)
            db.session.commit()
            backtest = Backtest(
                stock_id=999999,
                strategy_id=strategy.id,
                start_date=pd.Timestamp("2024-01-01").date(),
                end_date=pd.Timestamp("2024-01-03").date(),
                initial_capital=100000,
                status="pending",
            )
            db.session.add(backtest)
            db.session.commit()

            result = run_backtest_task.run(backtest.id)

            assert result == {
                "status": "failed",
                "error": "回测任务失败，请查看服务端日志。",
                "backtest_id": backtest.id,
            }
            assert db.session.get(Backtest, backtest.id).status == "failed"
        finally:
            db.session.remove()
            db.drop_all()


def test_backtest_api_rejects_zero_initial_capital(monkeypatch):
    monkeypatch.setattr(routes, "data_manager", _BacktestDataManager())
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        try:
            client = app.test_client()
            response = client.post(
                "/api/backtest",
                json={
                    "stock_code": "399998",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "initial_capital": 0,
                },
            )

            assert response.status_code == 400
            assert response.get_json() == {"error": "初始资金必须大于 0"}
        finally:
            db.session.remove()
            db.drop_all()


def test_backtest_api_marks_failed_status_after_worker_error(monkeypatch):
    monkeypatch.setattr(routes, "data_manager", _EmptyBacktestDataManager())
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        try:
            client = app.test_client()
            response = client.post(
                "/api/backtest",
                json={
                    "stock_code": "399998",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "initial_capital": 100000,
                },
            )

            assert response.status_code == 200
            payload = response.get_json()
            status = client.get(f"/api/task/{payload['task_id']}").get_json()

            assert status == {
                "state": "FAILURE",
                "error": "回测失败，请查看服务端日志。",
            }
            db.session.remove()
            assert db.session.get(Backtest, payload["backtest_id"]).status == "failed"
        finally:
            db.session.remove()
            db.drop_all()


def test_backtest_api_rejects_unsupported_strategy(monkeypatch):
    monkeypatch.setattr(routes, "data_manager", _BacktestDataManager())
    app.config["TESTING"] = True
    with app.app_context():
        db.create_all()
        try:
            strategy = routes.Strategy(
                name="技术分析策略",
                description="测试用策略",
                type="technical",
                parameters={},
                is_active=True,
            )
            db.session.add(strategy)
            db.session.commit()

            client = app.test_client()
            response = client.post(
                "/api/backtest",
                json={
                    "stock_code": "399998",
                    "strategy_id": strategy.id,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "initial_capital": 100000,
                },
            )

            assert response.status_code == 400
            assert response.get_json() == {
                "error": "当前 Web 回测仅支持买入持有基线策略"
            }
        finally:
            db.session.remove()
            db.drop_all()
