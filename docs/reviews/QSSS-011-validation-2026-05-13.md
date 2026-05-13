# QSSS-011 Web 回测占位功能跑通

## 任务范围

- 任务 ID: QSSS-011
- 目标: 将 Web 回测 API 从 501 占位提升为最小可执行回测链路，复用真实日线数据，写入 Backtest 记录并返回可展示指标。
- 日期: 2026-05-13

## 修复项

- 新增 `QuantStrategy.backtest()`，用真实日线 `close` 计算买入持有基线指标。
- `/api/backtest` 改为创建 Backtest 记录并启动回测任务。
- `/api/task/<uuid>` 增加内存级回测任务状态查询，开发环境无 Celery worker 也可展示结果。
- `perform_backtest` Celery 任务改为复用同一回测核心逻辑。
- `web/tasks.py` 修正 Backtest JSON 字段写入为 `results`。
- `web/README.md` 更新回测 API 说明，不再描述为 501 占位。

## 验证结果

| 验证项 | 命令 | 退出码 | 结果摘要 |
| --- | --- | --- | --- |
| 相关测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with Flask --with Flask-SQLAlchemy --with Flask-Migrate --with Celery --with redis pytest -q tests/test_strategy_core.py tests/test_web_tasks.py tests/test_web_mvp_pages.py` | 0 | `28 passed` |
| flake8 | `uv run --with flake8 flake8 src tests web scripts main.py` | 0 | 无输出 |
| Web 回测 HTTP smoke | `python3` 标准库请求 `POST /api/backtest` 并轮询 `/api/task/<uuid>` | 0 | `POST /api/backtest` 返回 200；任务最终 `SUCCESS`；返回 `total_return`、`annual_return`、`max_drawdown`、`sharpe_ratio`、`win_rate`、`total_trades` |

## 结论

Web 回测不再是 501 占位。当前实现是最小买入持有基线回测，可验证真实数据链路、任务状态链路和结果展示所需字段。
