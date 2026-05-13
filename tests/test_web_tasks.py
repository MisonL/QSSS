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

from web.tasks import _resolve_stock_info


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


def test_resolve_stock_info_uses_data_manager_metadata():
    stock_info = _resolve_stock_info(_StockListDataManager(), "600519")

    assert stock_info == {"symbol": "600519", "name": "贵州茅台", "market": "SH"}


def test_resolve_stock_info_falls_back_when_metadata_unavailable():
    stock_info = _resolve_stock_info(_FailingStockListDataManager(), "600519")

    assert stock_info == {"symbol": "600519", "name": "600519", "market": "Unknown"}
