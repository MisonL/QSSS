"""Tests for DataManager's data source health monitoring and dynamic降级逻辑.

These tests are purely in-memory and do not touch any real external
adapters or networks. They verify that:

- 连续失败会被记录，并在达到阈值后进入“降级窗口”；
- 当主数据源处于降级状态时，get_daily_data / get_stock_list
  会优先尝试备用数据源；
- 在这种降级场景下，主数据源不会被多余调用，从而减少整体延迟。
"""

from __future__ import annotations

import time

import pandas as pd

from qsss.data.manager import DataManager


class _FailingAdapter:
    """简单失败型适配器：所有调用都会抛异常。"""

    def __init__(self) -> None:
        self.daily_calls = 0
        self.list_calls = 0

    def get_daily_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:  # noqa: D401
        self.daily_calls += 1
        raise RuntimeError("forced failure from _FailingAdapter")

    def get_stock_list(self) -> pd.DataFrame:  # noqa: D401
        self.list_calls += 1
        raise RuntimeError("forced failure from _FailingAdapter")


class _SuccessAdapter:
    """成功型适配器：返回带有 `source` 标记的小 DataFrame。"""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    def get_daily_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:  # noqa: D401
        return pd.DataFrame(
            [
                {"symbol": symbol, "source": self.source_name},
            ]
        )

    def get_stock_list(self) -> pd.DataFrame:  # noqa: D401
        return pd.DataFrame(
            [
                {"symbol": "000001", "name": "测试股票", "market": self.source_name},
            ]
        )


class _DummyManager(DataManager):
    """轻量级 DataManager，用于避免在测试中初始化真实外部适配器。

    注意：这里不会调用 DataManager.__init__，而是手工设置所需字段，
    以便专注测试健康监控与主/备源切换逻辑。
    """

    def __init__(self) -> None:  # noqa: D401
        # 模拟适配器注册表与主/备数据源配置
        self.adapters = {}
        self.primary_source = "pytdx"
        self.backup_source = "baostock"

        # 健康监控相关字段
        self._source_health = {}
        # 对于测试，希望快速触发降级，因此把阈值设为 1
        self.health_failure_threshold = 1
        # 冷却时间设置得较长，确保测试过程中处于降级窗口
        self.health_cooldown = 60.0


def test_mark_failure_sets_degraded_window() -> None:
    """连续失败达到阈值后，数据源应进入降级状态。"""
    mgr = _DummyManager()

    assert not mgr._is_degraded("pytdx")

    before = time.time()
    mgr._mark_failure("pytdx")

    health = mgr.get_source_health()["pytdx"]
    assert health["consecutive_failures"] == 1
    assert health["last_error_ts"] >= before
    # 阈值为 1，故一次失败即应进入降级窗口
    assert mgr._is_degraded("pytdx")
    assert health["degraded_until"] > time.time()


def test_degraded_primary_skips_to_backup_for_daily_data() -> None:
    """当主数据源处于降级状态时，应优先调用备用数据源获取日线数据。"""
    mgr = _DummyManager()

    failing = _FailingAdapter()
    backup = _SuccessAdapter("baostock")

    mgr.adapters = {
        "pytdx": failing,
        "baostock": backup,
    }

    # 主数据源先被标记为降级
    mgr._mark_failure("pytdx")
    assert mgr._is_degraded("pytdx")

    df = mgr.get_daily_data("000001", "20220101", "20220131")

    # 结果应来自备用数据源，且不为空
    assert not df.empty
    assert set(df["source"]) == {"baostock"}

    # 在降级窗口内，主数据源不应被多余调用
    assert failing.daily_calls == 0


def test_degraded_primary_skips_to_backup_for_stock_list() -> None:
    """当主数据源处于降级状态时，应优先调用备用数据源获取股票列表。"""
    mgr = _DummyManager()

    failing = _FailingAdapter()
    backup = _SuccessAdapter("baostock")

    mgr.adapters = {
        "pytdx": failing,
        "baostock": backup,
    }

    mgr._mark_failure("pytdx")
    assert mgr._is_degraded("pytdx")

    df = mgr.get_stock_list()

    assert not df.empty
    # market 字段中会塞入我们在 _SuccessAdapter 中写入的 source_name
    assert set(df["market"]) == {"baostock"}
    assert failing.list_calls == 0
