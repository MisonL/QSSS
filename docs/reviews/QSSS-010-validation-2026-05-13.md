# QSSS-010 全面功能测试与验证收口

## 任务范围

- 任务 ID: QSSS-010
- 目标: 对当前仓库已实现功能执行全面测试矩阵，覆盖包导入、依赖契约、数据源契约、策略、缓存、SQLite schema、CLI、日志分析、Web 配置、Web 页面和任务接口。
- 日期: 2026-05-13

## 修复项

- 修复 `scripts/analyze_logs.py` 直接执行时找不到 `src` 包的问题。
- 修复 `main.py`、`web/routes.py`、`web/tasks.py` 和 Web 测试文件的 flake8 问题。
- 修复 pandas 当前版本不再支持 `fillna(method=...)` 导致策略分析失败的问题。
- 修复 `MLModel.train_model()` 对整张 DataFrame `dropna()`，导致 pytdx 空辅助列清空训练集的问题。
- 将 LightGBM 导入失败从模块导入期崩溃改为训练期显式失败，避免无关特征准备路径被 `libomp` 缺失阻断。
- 补齐 `/stocks`、`/strategies`、`/backtests`、`/stock/<id>` 模板，并新增 `/backtest` 占位页路由。

## 验证结果

| 验证项 | 命令 | 退出码 | 结果摘要 |
| --- | --- | --- | --- |
| 默认非 external 测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q` | 0 | `53 passed, 5 skipped, 10 deselected` |
| Web 定向测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with Flask --with Flask-SQLAlchemy --with Flask-Migrate --with Celery --with redis pytest -q tests/test_web_config.py tests/test_web_mvp_pages.py tests/test_web_tasks.py` | 0 | `25 passed` |
| 相关策略回归 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_strategy_core.py tests/test_cli_analyze_contract.py` | 0 | `11 passed` |
| flake8 | `uv run --with flake8 flake8 src tests web scripts main.py` | 0 | 无输出 |
| 装饰字符扫描 | `rg -n -P "<emoji-and-box-drawing-regex>" README.md web/README.md scripts src/qsss/interactive_cli.py src/qsss/simple_interactive.py web/run.py web/templates` | 1 | 无命中 |
| LightGBM 导入 | `uv run --with lightgbm python -c "import lightgbm; print(lightgbm.__version__)"` | 0 | `4.6.0` |
| CLI version | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm qsss version` | 0 | `QSSS版本: 2.0.0` |
| CLI config | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm qsss config` | 0 | 输出主数据源、线程、阈值和降级配置 |
| CLI sources | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm qsss sources` | 0 | 注册 `akshare` 和 `pytdx` |
| CLI analyze smoke | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm qsss analyze --start-date 20250101 --limit 1 --output /tmp/qsss_analyze_smoke.csv` | 0 | pytdx 获取 4946 只股票；样本日线 364 条；LightGBM 训练完成；单股分析成功 1、失败 0；筛选结果为 0 |
| SQLite 初始化脚本 | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm python scripts/init_sqlite.py /tmp/qsss_smoke.sqlite` | 0 | `SQLite schema initialized: /tmp/qsss_smoke.sqlite` |
| 日志分析脚本 | `uv run --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm python scripts/analyze_logs.py --help` | 0 | 正常显示帮助 |
| Web HTTP 页面 smoke | `python3` 标准库请求 `http://127.0.0.1:5055` 页面列表 | 0 | `/`, `/stocks`, `/strategies`, `/analysis`, `/market`, `/boards`, `/watchlists/ai-tech`, `/backtest`, `/backtests` 均返回 200 |
| Web API smoke | `python3` 标准库请求 `/api/task/not-a-uuid` 和 `POST /api/backtest` | 0 | 非法 task ID 返回 400；回测 API 已在 QSSS-011 中跑通为 200/SUCCESS |
| 浏览器渲染检查 | Chrome DevTools snapshot | 0 | `/stocks` 和 `/backtest` 页面可渲染并暴露主要交互元素 |
| external 数据源测试 | `uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with akshare --with baostock --with tushare pytest -q -m external tests/test_data_adapters_external.py -rs --tb=short` | 0 | `6 passed, 4 skipped`；Tushare 因未配置 token 跳过；Baostock 实时行情明确不支持并跳过 |

## 外部依赖记录

- macOS 本机缺少 `libomp.dylib` 时，LightGBM 导入失败。已执行 `brew install libomp`，安装版本为 `/usr/local/Cellar/libomp/22.1.5`。
- AKShare smoke 中出现过上游连接中断，Web 页面按设计脱敏展示错误并返回 200。
- Tushare 未配置 token，external 测试按契约跳过，不宣称真实 Tushare 已跑通。
- Baostock 股票列表和日线 external 测试已真实通过；实时行情接口明确不支持，不宣称实时行情已跑通。

## 结论

当前仓库默认自动化测试、Web 定向测试、CLI smoke、Web HTTP smoke、SQLite 初始化、日志脚本、LightGBM 导入、静态 lint 和装饰字符扫描均已通过。外部数据源中 AKShare、pytdx、Baostock 股票列表和 Baostock 日线有真实契约通过；Tushare 受凭据限制未覆盖真实成功，Baostock 实时行情明确不支持并跳过。Web 回测已在 QSSS-011 中从占位提升为可执行 MVP。
