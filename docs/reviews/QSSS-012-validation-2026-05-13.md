# QSSS-012 Tushare 离线契约覆盖

## 任务范围

- 任务 ID: QSSS-012
- 目标: 在缺少 Tushare token 的本机环境下，为 TushareAdapter 增加离线字段契约测试，覆盖股票列表、日线和实时行情字段映射。
- 日期: 2026-05-13

## 修复项

- 新增 `tests/test_tushare_adapter_contract.py`，通过 fake Tushare 客户端验证字段映射。
- 覆盖 `get_stock_list()` 的市场映射、`get_daily_data()` 的标准字段输出、`get_realtime_data()` 的实时字段对齐。
- 明确当前本机 `QSSS_TUSHARE_TOKEN` 和 `TUSHARE_TOKEN` 均未设置，真实 external 仍不能宣称通过。

## 验证结果

| 验证项 | 命令 | 退出码 | 结果摘要 |
| --- | --- | --- | --- |
| 离线契约测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_tushare_adapter_contract.py tests/test_baostock_adapter_contract.py` | 0 | `6 passed` |
| 默认非 external 测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q` | 0 | `53 passed, 5 skipped, 10 deselected` |
| external 数据源测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with akshare --with baostock --with tushare pytest -q -m external tests/test_data_adapters_external.py -rs --tb=short` | 0 | `6 passed, 4 skipped`；Tushare 三项因 token 缺失跳过 |
| 环境 token 检查 | `python3` 检查 `QSSS_TUSHARE_TOKEN` 和 `TUSHARE_TOKEN` | 0 | 两者均为 `missing` |
| flake8 | `uv run --with flake8 flake8 src tests web scripts main.py` | 0 | 无输出 |

## 结论

TushareAdapter 的本地字段转换契约已覆盖，但真实 Tushare external 链路仍需要有效 token。当前不能把 Tushare live 访问宣称为已跑通。
