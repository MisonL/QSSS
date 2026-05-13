# QSSS-013 Baostock 实时行情能力边界显式化

## 任务范围

- 任务 ID: QSSS-013
- 目标: 将 BaostockAdapter.get_realtime_data 从静默空结果改为显式不支持，避免把未实现能力误判为上游空数据。
- 日期: 2026-05-13

## 修复项

- `BaostockAdapter.get_realtime_data()` 改为抛出 `NotImplementedError`。
- external 测试捕获 `NotImplementedError` 并显式 skip。
- Baostock 离线契约测试新增实时行情不支持边界断言。

## 验证结果

| 验证项 | 命令 | 退出码 | 结果摘要 |
| --- | --- | --- | --- |
| Baostock 契约测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_baostock_adapter_contract.py` | 0 | `3 passed` |
| external 数据源测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with akshare --with baostock --with tushare pytest -q -m external tests/test_data_adapters_external.py -rs --tb=short` | 0 | `6 passed, 4 skipped`；Baostock 实时行情明确 skip 为不支持 |
| flake8 | `uv run --with flake8 flake8 src tests web scripts main.py` | 0 | 无输出 |

## 结论

Baostock 的支持边界已明确为股票列表和日线数据，不再把实时行情误报为空数据。默认实时行情链路仍由 pytdx 承担。
