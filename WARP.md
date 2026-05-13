# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## 关键命令（Key commands）

### 环境准备（Environment setup）
- 使用 uv 创建并激活虚拟环境：
  - `uv venv`
  - `source .venv/bin/activate`  （macOS/Linux）
- 安装核心依赖（CLI / 引擎）：
  - `uv pip install -r requirements.txt`
- 安装完整开发依赖（包含测试、格式化、类型检查，来源于 `pyproject.toml`）：
  - `pip install .[dev]`
    （会安装 `pytest`、`pytest-cov`、`black`、`isort`、`flake8`、`mypy` 等工具）

### CLI / 核心引擎（core engine）
- 以可编辑模式安装包，提供 `qsss` 控制台命令：
  - `pip install -e .`
- 启动默认 CLI（无子命令时进入简化交互式 TUI）：
  - `qsss`
- 启动完整交互式菜单：
  - `qsss interactive`
- 从命令行执行标准全市场分析：
  - `qsss analyze`
  - 常用参数示例：
    - `qsss analyze --start-date 20220101 --limit 100`
    - `qsss analyze --output results.csv --limit 50`
- 查看版本和当前配置：
  - `qsss version`
  - `qsss config`

### 交互式演示脚本（Interactive demo scripts）
适合做手工 QA 和复现交互/体验相关问题（均在项目根目录、已激活虚拟环境下运行）：
- 演示流程：
  - `python scripts/qsss_demo.py`
- 改进版完整交互菜单：
  - `python scripts/improved_interactive.py`
- 旧版完整交互菜单：
  - `python scripts/interactive_qsss.py`

### Web 应用（web application）
除特别说明外，以下命令都在 `web/` 目录中执行：
- 安装 Web 端依赖：
  - `cd web`
  - `pip install -r requirements.txt`
- 初始化数据库：
  - `python run.py --init-db`
- 启动 Flask 应用：
  - 开发环境：`python run.py --env dev`
  - 生产风格：`python run.py --env prod`
- 启动 Celery Worker（异步分析 / 回测任务，需要本机 Redis）：
  - `celery -A app.celery worker --loglevel=info`

### 分布式 / 高性能执行（Distributed / high‑performance execution）
核心策略类本身已经在单进程内做了多线程并行；如果需要真实的跨进程 / 多节点分布式：
- 启动一个或多个分布式 Worker：
  - `python -m src.qsss.distributed.worker --worker-id=node1 --max-workers=4`
  - `python -m src.qsss.distributed.worker --worker-id=node2 --max-workers=4`
- 在 Python 代码中使用优化版策略（带多级缓存和分布式调度）：
  - 参考 `src/qsss/core/optimized_strategy.py` 中 `OptimizedQuantStrategy` 的上下文管理器用法。

### 测试（Tests）
测试文件主要位于 `tests/` 目录下，例如 `tests/test_strategy_core.py` 等：
- 运行全部测试：
  - `pytest`
- 只运行单个文件：
  - `pytest tests/test_strategy_core.py`
- 只运行单个测试用例：
  - `pytest tests/test_strategy_core.py::test_apply_filters_keeps_valid_row_and_computes_total_score`

注意：未来如果为数据适配器编写集成测试，这类测试会访问真实外部数据源（pytdx / Tushare / Baostock），
网络或上游服务异常时可能导致非功能性失败。

### 代码格式 / Lint / 类型检查
工具配置见 `pyproject.toml`，包括：`black`、`isort`、`flake8`、`mypy`。
- 代码格式化：
  - `black src tests`  （或直接 `black .`）
  - `isort src tests`
- Lint 检查：
  - `flake8 src tests`
- 静态类型检查：
  - `mypy src`

### 性能脚本（Performance scripts）
- 运行提供的性能基准并查看结果：
  - `python scripts/performance_test.py`
  - `cat performance_test_results.json`

## 高层架构（High-level architecture）

### 总体概览（Overview）
QSSS（Quantitative Stock Selection System）是一个面向 A 股市场的高性能量化选股引擎，整体由以下几层组成：
- 核心 **量化策略引擎**：位于 `src/qsss/`，负责特征工程、机器学习模型、多因子评分、过滤条件以及性能监控。
- 多种前端：
  - CLI（`qsss` 控制台命令 + 交互式 CLI 界面）。
  - Web UI（`web/` 下的 Flask 应用 + Celery Worker + SQLAlchemy 模型）。
- 数据访问层：基于 `DataAdapter` + `DataManager` 的多数据源抽象，当前支持 pytdx / Tushare / Baostock，带连接池与内存缓存。
- 可选的 **分布式执行与分布式缓存**：通过 Redis / Celery 与自定义 Worker 进程实现。

典型数据/控制流：
1. 数据层通过 `DataManager` 调用已注册的数据源适配器（如 `PytdxAdapter` / `TushareAdapter` / `BaostockAdapter`）拉取并标准化 A 股历史行情（价格 / 成交量等）。
2. 特征与机器学习层构建各类因子与技术指标特征，训练 LightGBM 分类模型，预测未来 5 日上涨概率。
3. 策略层将 ML 输出、动量、短线“爆发”信号、波动率、成交量及风险过滤条件综合为总评分 `total_score`。
4. 前端层（CLI / 交互式 / Web+Celery / 分布式 Worker）编排批量分析、持久化和展示（表格、页面、回测结果等）。

### 核心引擎目录结构（`src/qsss/`）

#### `cli.py` / `interactive_cli.py` / `simple_interactive.py`
- `cli.py`：定义 `qsss` 的 Click 入口点：
  - 直接运行 `qsss`：启动 **简化版交互式 CLI**（`SimpleInteractiveCLI`）。
  - `qsss interactive`：启动基于 Rich 的 **完整交互菜单**（`InteractiveCLI`）。
  - `qsss analyze`：调用 `QuantStrategy.run_analysis` 执行全市场分析，并以表格形式输出；可选写入 CSV。
  - `qsss version` / `qsss config`：查看版本和当前配置。
- `interactive_cli.py`：实现 Rich 风格 TUI：
  - 包含“运行分析”“查看上次结果”“系统配置”“策略参数”等菜单。
  - 在运行分析时展示进度指示、允许设置输出数量和是否保存结果。
- `simple_interactive.py`：更鲁棒、依赖更少的简化交互界面：
  - Rich 不可用时自动退化为纯文本交互。
  - 保留主菜单、基础配置展示和结果浏览能力。

这些 CLI 是复现用户可见行为（问题重现 / 手工验证）的首选入口，无需触碰 Web 或分布式栈。

#### `core/strategy.py` 与 `core/optimized_strategy.py`
- `QuantStrategy`（`strategy.py`）是 **标准单进程策略引擎**，CLI 和交互式界面默认使用：
  - 通过 `data_manager.get_stock_list()` 获取股票池，并用正则过滤保留主板、科创板、创业板。
  - 调用 `data_manager.get_daily_data` 获取日线数据，内部带重试与简单缓存。
  - 将数据交给 `TechnicalAnalyzer` 与 `ShortTermAnalyzer` 计算技术指标与短线爆发行评分。
  - 使用 `MLModel` 为每只股票训练 LightGBM，并得到 5 日上涨概率预测。
  - 用 `settings` 中的阈值（上涨概率、动量、RSI、波动率、成交量、价格等）做多重过滤，并计算综合得分 `total_score`。
  - 使用 `PerformanceMonitor.get_optimal_thread_count` 自适应选择线程数，在 `ThreadPoolExecutor` 中并行逐股分析。
  - 维护价格历史与 MA15 的内存缓存，并通过 `calculate_ma15`、`get_analysis_summary` 对外暴露。
- `OptimizedQuantStrategy`（`optimized_strategy.py`）是 **更高级、面向缓存与分布式的策略引擎**：
  - 使用本地缓存 `cache_manager` 与分布式缓存 `DistributedCache`（Redis）做多级缓存，加速股票列表、单股历史和分析结果。
  - 借助 `TaskScheduler` 与 `DistributedCache`，将工作拆分成批量任务并分发到 Worker 进程 / 节点。
  - 提供 `_run_local_parallel_analysis` 和 `_run_distributed_batch_analysis` 两种执行路径。
  - 通过上下文管理器自动管理 `ProcessPoolExecutor` 与 `ThreadPoolExecutor` 生命周期，适合在高性能脚本中以 `with OptimizedQuantStrategy(...)` 使用。

通常：
- 想对 CLI 行为进行修改或对比：优先改动 `QuantStrategy`。
- 有明确的性能 / 扩展要求（缓存命中率、分布式调度等）：再考虑使用或扩展 `OptimizedQuantStrategy`。

#### `data/` 数据层
- `adapters.py`：
  - 定义抽象基类 `DataAdapter` 与默认实现 `PytdxAdapter`。
  - 负责：
    - 通过 pytdx 拉取股票列表与日线数据；
    - 根据代码前缀判断交易所 / 板块（`60/68`→上交所主板/科创板，`00/30`→深交所主板/创业板，`83/87/43`→北交所等）；
    - 维护 pytdx 连接池，支持最佳 IP 选择与备用服务器列表；
    - 做简单的内存级 TTL 缓存，并定期清理过期条目。
- `data/sources/`：
  - `tushare_adapter.py`：基于 Tushare Pro 的可选数据源，需要配置 Token。
  - `baostock_adapter.py`：基于 baostock 的可选历史数据源。
- `manager.py`：
  - 提供全局单例 `data_manager`。
  - 在初始化时自动注册可用的数据源（pytdx 默认 + 可用的 Tushare / Baostock）。
  - 屏蔽具体数据源，仅暴露 `get_stock_list` / `get_daily_data` / `get_realtime_data` 等方法。
  - 为缺省日期范围提供合理默认值（通常是最近一年数据）。

所有与数据源替换、Mock、数据获取策略相关的改动都应优先经由 `DataManager` 和 `DataAdapter`。

#### `config/settings.py`
- 使用 `pydantic_settings.BaseSettings` 定义 `Settings` 配置类，并创建全局实例 `settings`。
- 主要职责：
- 数据源：通过 `primary_data_source` / `backup_data_source` 字段以及环境变量 `QSSS_PRIMARY_DATA_SOURCE` / `QSSS_BACKUP_DATA_SOURCE` 控制，
  默认使用 `pytdx`，可选启用 `tushare`、`baostock` 等适配器（前提是正确安装并配置凭据）。
  - 性能参数：`min_workers` / `max_workers`、CPU / 内存阈值等。
  - 缓存：是否启用缓存、缓存目录、TTL 等。
  - 选股条件：最小预测概率、动量下限、RSI 区间、最大波动率、最小成交量 / 价格、最小数据天数等。
  - 日志与重试策略：默认日志级别、日志文件路径、重试次数和延迟。
- 所有字段都可以通过 `.env` 或环境变量（前缀 `QSSS_`）覆盖。

修改筛选逻辑或性能参数时，优先更新 `Settings` 默认值或其消费者，而不是在代码里硬编码常量。

#### `models/ml_model.py`
- 封装 **所有与 ML 特征工程与模型训练相关的逻辑**：
  - 计算多周期动量（1M/3M/6M）、波动率、成交量比、RSI、MACD 及其信号线、短期均线等特征；
  - 对价格 / 成交量 / 换手率做异常值处理与填充；
  - 构造 5 日前瞻收益率作为目标，按是否大于均值二值化得到分类标签；
  - 使用 `StandardScaler` 标准化特征，并按 `settings.ml_*` 参数划分训练 / 测试集；
  - 训练 `lightgbm.LGBMClassifier`，记录训练集与测试集准确率；
  - `predict` 方法对最新一根 K 线返回上涨概率，用于后续评分。

如需调整特征或模型结构，需要同步更新 `features` 列表以及下游依赖该输出的策略逻辑。

#### 其他核心模块：`strategies/`、`utils/`、`cache/`、`database/`
- `strategies/technical.py` / `short_term.py`：
  - 负责技术因子计算、MACD 信号识别和短线“爆发潜力”评分等。
- `utils/performance.py`：
  - `PerformanceMonitor` 提供系统负载监控与自适应线程数计算，并记录整体耗时、峰值 CPU/内存等统计信息。
- `cache/manager.py` 与 `distributed/cache.py`：
  - 前者提供本地缓存管理（包含统计与按模式清理），后者包装 Redis 或内存字典实现分布式缓存，支持 TTL、命中率统计等。
- `database/optimizer.py`：
  - 封装数据库层面的优化（如索引、查询调优），主要用于较重的 Web / 分布式部署场景。

### Web 应用（`web/`）
Web 应用是在核心引擎之上的独立层，但与 `src/qsss/` 紧密耦合：

- `web/app.py`：
  - 创建 Flask 应用、SQLAlchemy `db`、Alembic `Migrate` 以及 Celery 实例；
  - 通过环境变量 `DATABASE_URL`、`REDIS_URL`、`SECRET_KEY` 配置数据库与 Redis；
  - 导入并注册 `models` 与 `routes`。
- `web/models.py`：
  - 定义关系型模型：`Stock`、`Strategy`、`Backtest`、`AnalysisResult`；
  - 使用 JSON 字段存储策略参数、回测结果及丰富的分析输出；
  - 所有模型都提供 `to_dict()` 方便模板与 API 使用。
- `web/routes.py`：
  - 实现 HTML 视图：股票列表 / 详情、策略列表、回测列表、分析页面等；
  - 提供 JSON API：
    - `POST /api/analyze`：提交分析任务到 Celery（`perform_analysis`）；
    - `GET /api/task/<task_id>`：轮询 Celery 任务状态；
    - `POST /api/backtest`：创建 `Backtest` 记录并触发 `perform_backtest` 任务；
  - Celery 任务内部会调用核心引擎（`QuantStrategy` + `DataManager`），并把结果写入 `AnalysisResult` / `Backtest`。
- `web/config.py` 与 `web/README.md`：
  - 描述开发 / 测试 / 生产多环境配置；
  - 给出数据库与 Redis 默认连接、日志路径、上传目录、分页大小、缓存类型以及 Cookie 安全策略等。

当修改会影响 CLI 与 Web 共同行为的逻辑（例如评分公式、选股过滤规则）时，应优先调整核心模块（`src/qsss/core`、`src/qsss/models`、`src/qsss/strategies`），保持两端行为一致。

### 分布式处理与分布式缓存（`src/qsss/distributed/`）
- `scheduler.py`（`TaskScheduler`）：
  - 负责任务创建、持久化与状态维护；
  - 如 Redis 可用，则使用有序集合 + 哈希存储任务队列与元数据；否则回退到内存队列；
  - `distribute_stock_analysis` 会把股票列表拆分成批次任务，按优先级入队；
  - `get_cluster_status` 使用 `psutil` 提供队列长度、正在运行任务数及 CPU / 内存使用情况。
- `worker.py`（`DistributedWorker`）：
  - 独立的长生命周期 Worker 进程，轮询 `TaskScheduler.get_pending_tasks`；
  - 每个任务通常是一个批次（`stock_analysis_batch`），内部再用线程池并行调用 `QuantStrategy.analyze_single_stock`；
  - 根据本机资源使用情况决定是否接收新任务，并周期性输出健康状态统计；
  - 在 Celery 可用时，还会注册 Celery 任务封装同样的逻辑。
- `cache.py`（`DistributedCache`）：
  - 基于 Redis（或内存回退）的通用分布式缓存；
  - 使用 pickle 为主、JSON 为辅进行序列化；
  - 提供 `get` / `set` / `delete` / `exists` / `clear_pattern` / `get_stats` 等统一接口，用于缓存 DataFrame 等 Python 对象。

这些模块主要在需要扩展到多进程 / 多机器、以及调优缓存命中率与队列吞吐时才需要深入修改。单机场景下一般仅需使用 `QuantStrategy` 的内建多线程能力即可。
