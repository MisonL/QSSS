# QSSS-014 修复回测审查问题

## 任务范围

- 任务 ID: QSSS-014
- 目标: 修复 Code Review 后确认有效的问题，覆盖回测资金校验、Web 测试数据库隔离、回测策略语义和历史文档口径。
- 日期: 2026-05-13

## 修复项

- `initial_capital=0` 不再被 `or 100000.0` 静默替换，Web API 返回 400，策略核心返回显式错误。
- pytest 进程默认设置 `QSSS_WEB_ENV=testing`，Web app 导入期使用 `sqlite:///:memory:`，相关测试补充隔离断言和清理。
- `/backtest` 页面只展示买入持有基线，服务端拒绝非 `buy_hold` 策略 ID，避免把技术分析、机器学习或短线策略伪装成已实现回测。
- `web.tasks.run_backtest_task()` 执行层同步拒绝缺失股票和非 `buy_hold` 策略，避免旧 Celery 入口绕过 API 创建校验。
- 回测 worker 和 Celery 失败路径在 `rollback()` 后重新按 ID 读取 `Backtest` 再标记失败，避免在回滚后的旧 ORM 实例上写状态。
- `/strategies` 页面展示策略参数前会脱敏 `token`、`api_key`、`secret`、`password` 等敏感键，避免数据库参数泄露到页面。
- `/backtests` 和股票详情页为缺失状态、日期提供占位展示。
- 清理 Web 导航和文档中的旧占位路由、未实现文案，并同步 Baostock 实时行情为显式不支持。

## 验证结果

| 验证项 | 命令 | 退出码 | 结果摘要 |
| --- | --- | --- | --- |
| Web 和策略定向测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with Flask --with Flask-SQLAlchemy --with Flask-Migrate --with Celery --with redis pytest -q tests/test_strategy_core.py tests/test_web_tasks.py tests/test_web_mvp_pages.py tests/test_web_config.py` | 0 | `35 passed` |
| 默认非 external 测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q` | 0 | `53 passed, 5 skipped, 10 deselected` |
| flake8 | `uv run --with flake8 flake8 src tests web scripts main.py` | 0 | 无输出 |
| diff 空白检查 | `git diff --check` | 0 | 无输出 |
| 过期文案扫描 | `rg -n "<stale-backtest-placeholder-regex>" web tests src docs issues.csv` | 1 | 无命中，`rg` 退出码 1 表示未匹配 |
| Web 页面 smoke | Chrome DevTools 打开 `http://127.0.0.1:5055/backtest` 并点击 `回测结果` | 0 | `/backtest` 展示买入持有基线；导航跳转 `/backtests` 返回 200 |
| Web API smoke | `python3` 标准库请求 `POST /api/backtest`，`initial_capital=0` | 0 | HTTP 400，响应 `{"error":"初始资金必须大于 0"}` |

## CodeRabbit 复审

最终复跑 `coderabbit review --prompt-only -t uncommitted`，结果为 `No findings`。

## 结论

本轮审查问题已修复并完成可复现验证。当前 Web 回测明确限定为买入持有基线，测试不再依赖开发库残留表，非法初始资金走显式错误路径。
