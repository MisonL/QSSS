# QSSS 全功能验证收口 - 2026-05-13

## 目标

- 对当前仓库已实现功能做全面测试，确认主要功能入口全部可运行。
- 覆盖包导入、依赖契约、数据源契约、策略、缓存、SQLite schema、CLI、日志分析、Web 页面、Web 任务接口、回测 API、全市场分析 API、静态 lint 和装饰字符扫描。
- 对无法在当前本机真实验证的外部能力记录真实原因，不伪造通过。

## 本轮发现与修复

- `QuantStrategy.backtest()` 在真实回测样本出现负收益时，年化收益计算可能因非正价格链路触发复数转换异常。本轮新增非正收盘价显式错误边界，并用测试覆盖。
- `/api/analyze_market` 的 `limit` 原先只限制最终返回，不限制实际扫描数量，导致 `limit=1` smoke 仍触发全市场扫描。本轮改为传入核心策略实际限制扫描数量。
- `/api/analyze_market` 在筛选结果为空时原先返回失败。本轮改为成功返回空结果，避免把“无符合条件股票”误报为任务失败。

## 验证矩阵

| 验证项 | 命令 | 退出码 | 摘要 |
| --- | --- | --- | --- |
| 默认非 external 测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q` | 0 | `53 passed, 5 skipped, 10 deselected` |
| Web 定向测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with Flask --with Flask-SQLAlchemy --with Flask-Migrate --with Celery --with redis pytest -q tests/test_web_config.py tests/test_web_mvp_pages.py tests/test_web_tasks.py` | 0 | `25 passed` |
| external 数据源测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with akshare --with baostock --with tushare pytest -q -m external tests/test_data_adapters_external.py -rs --tb=short` | 0 | `6 passed, 4 skipped, 11 warnings` |
| flake8 | `uv run --with flake8 flake8 src tests web scripts main.py` | 0 | 无输出 |
| 装饰字符扫描 | `rg -n -P "<decorative-unicode-regex>" README.md web/README.md scripts src/qsss/interactive_cli.py src/qsss/simple_interactive.py web/run.py web/templates` | 1 | 无命中，`rg` 退出码 1 表示未匹配 |
| CLI smoke | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm qsss version/config/sources/analyze --start-date 20250101 --limit 1 --output /tmp/qsss_analyze_smoke.csv` | 0 | version/config/sources 正常；analyze 真实连接 pytdx，获取 4946 只股票，样本日线 364 条，模型训练完成，成功 1 只、失败 0 只 |
| SQLite 初始化 | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm python scripts/init_sqlite.py /tmp/qsss_smoke.sqlite` | 0 | `SQLite schema initialized: /tmp/qsss_smoke.sqlite` |
| 日志分析帮助 | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm python scripts/analyze_logs.py --help` | 0 | 帮助信息正常输出 |
| Web 页面 HTTP smoke | `python3` 标准库/requests 请求 `http://127.0.0.1:5055` 的 `/`、`/stocks`、`/strategies`、`/analysis`、`/market`、`/boards`、`/watchlists/ai-tech`、`/backtest`、`/backtests` | 0 | 全部返回 200 |
| Web 回测 API smoke | `POST /api/backtest` 后轮询 `/api/task/<uuid>` | 0 | 返回 `SUCCESS`，包含 `annual_return`、`total_return`、`max_drawdown`、`sharpe_ratio`、`win_rate`、`total_trades` |
| Web 全市场分析 API smoke | `POST /api/analyze_market {"limit": 1}` 后轮询 `/api/market_status/<job_id>` | 0 | 真实连接 pytdx；状态中 `total=1`、`processed=1`、`progress=1.0`，最终 `SUCCESS`，空筛选结果返回 `{"count": 0, "results": []}` |
| Web 非法任务 ID | `GET /api/task/not-a-uuid` | 0 | 返回 400，响应为 `{"error": "任务 ID 格式无效"}` |

## 外部能力边界

- AKShare、pytdx、Baostock 股票列表和 Baostock 日线在 external 测试中有真实通过项。
- 当前本机未配置 `QSSS_TUSHARE_TOKEN` 或 `TUSHARE_TOKEN`，Tushare 三项 external 测试按契约跳过，不能宣称真实 Tushare live 链路已跑通。
- Baostock 实时行情能力被显式标记为不支持，external 测试按契约跳过。
- Web `/boards` 页面 smoke 期间 AKShare 曾出现上游连接中断，但页面仍按设计返回 200 并展示可用数据或公开错误信息。

## 结论

当前仓库主要功能入口均已完成可复现验证。默认测试、Web 测试、external 数据源契约、CLI、SQLite 初始化、日志分析、Web 页面、回测 API、全市场分析 API、lint 和装饰字符扫描均已按当前环境跑通或明确记录真实跳过原因。
