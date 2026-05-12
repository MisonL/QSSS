"""核心选股策略单元测试

这些测试不依赖外部数据源，只验证 QuantStrategy.apply_filters
的打分公式和过滤逻辑是否与当前设计保持一致。
"""

import copy

import pandas as pd

from qsss.config.settings import settings
from qsss.core.strategy import QuantStrategy


def _make_base_row():
    """构造一行刚好满足/略高于所有阈值的基础数据。"""
    return {
        "symbol": "000001",
        "name": "测试股票",
        "market": "深交所-主板",
        # 概率、动量、爆发因子
        "prediction": settings.min_prediction_threshold + 0.1,
        "momentum_score": settings.min_momentum_score + 0.1,
        "explosion_score": 1.0,
        # 波动率取阈值的一半，保证远低于上限
        "volatility": min(settings.max_volatility / 2, 0.3),
        # RSI 取区间中点
        "rsi": (settings.rsi_range[0] + settings.rsi_range[1]) / 2,
        # 成交量、价格略高于下限
        "volume": settings.min_volume + 10_000,
        "close": settings.min_price + 1.0,
    }


def test_apply_filters_keeps_valid_row_and_computes_total_score():
    """基础行应通过所有过滤条件，且 total_score 计算公式固定。"""
    strategy = QuantStrategy()

    base = _make_base_row()
    df = pd.DataFrame([base])

    result = strategy.apply_filters(df.copy())

    # 应该只有这一行被保留
    assert len(result) == 1

    row = result.iloc[0]

    expected_score = (
        base["prediction"] * 0.3
        + base["momentum_score"] * 0.2
        + base["explosion_score"] * 0.35
        + (1 - base["volatility"]) * 0.15
    )

    assert "total_score" in result.columns
    # 浮点误差允许一个很小的误差范围
    assert abs(row["total_score"] - expected_score) < 1e-9


def test_apply_filters_rejects_rows_violating_each_condition():
    """分别破坏各个筛选条件时，行应被完全过滤掉。"""
    strategy = QuantStrategy()
    base = _make_base_row()

    def _check_rejected(mods):
        row = copy.deepcopy(base)
        row.update(mods)
        df = pd.DataFrame([row])
        result = strategy.apply_filters(df)
        assert result.empty

    # 预测概率过低
    _check_rejected({"prediction": settings.min_prediction_threshold - 1e-3})

    # 动量得分过低
    _check_rejected({"momentum_score": settings.min_momentum_score - 1e-3})

    # RSI 低于下限
    _check_rejected({"rsi": settings.rsi_range[0] - 1})

    # RSI 高于上限
    _check_rejected({"rsi": settings.rsi_range[1] + 1})

    # 波动率超过上限
    _check_rejected({"volatility": settings.max_volatility + 1e-3})

    # 成交量不足
    _check_rejected({"volume": max(settings.min_volume - 1, 0)})

    # 股价过低
    _check_rejected({"close": settings.min_price - 1e-3})
