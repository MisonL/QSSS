# QSSS - 量化选股系统 v2.0

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**QSSS (Quantitative Stock Selection System)**是专为A股市场设计的高性能量化交易系统，采用现代Python架构，支持CLI命令行和Web界面双重操作模式。

## 核心特性

- **高性能架构**: 分布式计算 + 多级缓存 + 并行处理
- **双模式操作**: CLI命令行 + Web可视化界面
- **AI驱动**: LightGBM机器学习模型预测5日收益率
- **实时分析**: 多因子模型 + 技术指标 + 超短线爆发策略
- **分布式处理**: Redis任务队列 + 多工作节点并行计算
- **智能缓存**: L1内存缓存 + L2 Redis缓存，命中率80%+
- **数据源健康监控**: 主备数据源自动降级 / 冷却机制，配合 CLI `qsss sources` 与日志分析脚本监控稳定性
- **回测功能（规划中）**: 已预留回测模型与接口骨架，未来将支持历史数据策略验证和性能评估

## 技术架构

```
+--===============+    +--===============+    +--===============+
|   CLI客户端     |    |   Web界面       |    |  分布式工作节点  |
+--=======+--=====+    +--=======+--=====+    +--=======+--=====+
          |                      |                      |
          +--====================+--====================+
                                 |
                    +--===========v=============+
                    |    核心策略引擎            |
                    |  - 多因子评分模型          |
                    |  - 机器学习预测            |
                    |  - 技术指标分析            |
                    +--===========+--===========+
                                  |
                    +--===========v=============+
                    |    数据管理层              |
                    |  - 多数据源适配（pytdx / Tushare / Baostock） |
                    |  - 多级缓存系统            |
                    |  - 数据库优化              |
                    +--===========+--===========+
                                  |
                    +--===========v=============+
                    |    基础设施层              |
                    |  - Redis分布式缓存         |
                    |  - SQLAlchemy数据库        |
                    |  - Celery任务队列          |
                    +--=========================+
```

## 快速开始

### 环境要求
- **Python**: 3.9+
- **Redis**: 5.0+ (用于分布式缓存和任务队列)
- **操作系统**: Windows 10/11, macOS 10.15+, Linux Ubuntu 18.04+

### 安装步骤

#### 1. 克隆项目
```bash
git clone https://github.com/MisonL/QSSS.git
cd QSSS
```

#### 2. 创建虚拟环境并安装依赖
```bash
# 使用UV安装（推荐）
uv venv
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 使用阿里源安装依赖
uv pip install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple/
```

#### 3. 启动Redis服务
```bash
# macOS (使用Homebrew)
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Windows
# 下载并安装Redis for Windows
# 启动 redis-server.exe
```

## 使用方式

### CLI命令行模式

```bash
# 基础分析
qsss analyze

# 指定起始日期和最大分析/输出数量
qsss analyze --start-date 20240101 --limit 100

# 保存结果
qsss analyze --output results.csv

# 查看版本
qsss version

# 显示配置与数据源健康
qsss config
qsss sources
```

### Python API模式

```python
from src.qsss.core.optimized_strategy import OptimizedQuantStrategy

# 启用高性能模式
with OptimizedQuantStrategy(use_distributed=True, use_cache=True) as strategy:
    results = strategy.run_analysis()
    print(f"分析完成，选中 {len(results)} 只股票")

    # 查看性能统计
    summary = strategy.get_analysis_summary()
    print(f"缓存命中率: {summary['cache_stats']['hit_rate']:.1%}")
```

### Web界面模式

```bash
# 启动 Web 服务（开发环境）
python web/run.py --env dev --host 0.0.0.0 --port 5000

# 启动 Web 服务（生产环境）
python web/run.py --env prod --host 0.0.0.0 --port 8080

# 启动 Celery Worker（分布式任务）
celery -A web.app.celery worker --loglevel=info
```
# 访问 http://localhost:5000
```

### 分布式部署

```bash
# 启动多个工作节点
python -m src.qsss.distributed.worker --worker-id=node1 --max-workers=4
python -m src.qsss.distributed.worker --worker-id=node2 --max-workers=4
```

## 选股策略

### 多因子评分模型

| 因子类别 | 权重 | 指标说明 |
|---------|------|----------|
| **上涨概率**| 30% | LightGBM预测未来5日上涨概率 |
| **动量得分**| 20% | 1M/3M/6M动量综合评分 |
| **爆发潜力**| 35% | 短线爆发潜力综合评分 |
| **风险控制**| 15% | 波动率、成交量等风险指标 |

### 技术指标体系
- **RSI**: 14日相对强弱指标 (30-75范围)
- **MACD**: 12/26/9日MACD金叉信号
- **布林带**: 20日布林带位置分析
- **成交量**: 相对20日均量比率
- **波动率**: 20日年化波动率 (<60%)

### 超短线爆发策略
- **成交量突增**: 相对均量1.5倍以上
- **换手率变化**: 相对均换手率倍数
- **MACD金叉**: 金叉预判和即将金叉
- **价格位置**: 相对近期高低点位置
- **短期动量**: 3日价格动量

## 项目结构

```
QSSS/
+-- src/qsss/                    # 核心模块包
|   +-- cli.py                  # 命令行接口
|   +-- core/                   # 核心策略引擎
|   |   +-- strategy.py         # 原始策略类
|   |   +-- optimized_strategy.py # 高性能优化策略
|   +-- data/                   # 数据管理层
|   |   +-- manager.py          # 数据管理器
|   |   +-- sources/            # 数据源适配器
|   +-- models/                 # 机器学习模型
|   +-- strategies/             # 策略实现
|   +-- distributed/            # 分布式计算
|   |   +-- scheduler.py        # 任务调度器
|   |   +-- worker.py           # 工作节点
|   |   +-- cache.py            # 分布式缓存
|   +-- cache/                  # 缓存管理
|   |   +-- manager.py          # 高性能缓存管理器
|   +-- database/               # 数据库优化
|   |   +-- optimizer.py        # 数据库查询优化器
|   +-- config/                 # 配置管理
+-- web/                        # Web应用
|   +-- app.py                  # Flask应用
|   +-- routes.py               # API路由
|   +-- tasks.py                # 异步任务
|   +-- templates/              # HTML模板
+-- scripts/                    # 工具脚本
|   +-- performance_test.py     # 性能测试脚本
+-- data/                       # 数据存储
+-- tests/                      # 测试文件
+-- pyproject.toml              # 项目配置
+-- requirements.txt            # 依赖列表
```

## 配置参数

### 系统配置（与 `src/qsss/config/settings.py` 对应）
```python
# 数据源配置（可通过环境变量 QSSS_PRIMARY_DATA_SOURCE / QSSS_BACKUP_DATA_SOURCE 覆盖）
primary_data_source = "pytdx"      # 主数据源（默认：pytdx）
backup_data_source = ""            # 备用数据源，可选："tushare"、"baostock" 等

# 性能与缓存
min_workers = 4                    # 最小线程数
max_workers = 10                   # 最大线程数
cpu_threshold = 75.0               # CPU 使用率阈值（%）
memory_threshold = 85.0            # 内存使用率阈值（%）
cache_enabled = True               # 是否启用缓存
cache_dir = "data/cache"           # 缓存目录
cache_ttl = 3600                   # 缓存过期时间（秒）

# 选股条件
min_prediction_threshold = 0.6     # 最小上涨概率
min_momentum_score = -0.1          # 最小动量得分
rsi_range = (30, 75)               # RSI 合法区间
max_volatility = 0.6               # 最大波动率
min_volume = 50_000                # 最小成交量（手）
min_price = 3.0                    # 最小股价（元）

# 数据要求
min_data_days = 120                # 单只股票最少需要的历史交易日数

# 数据源健康监控
datasource_failure_threshold = 3   # 单数据源连续失败多少次后进入降级
datasource_cooldown_seconds = 300  # 降级后的冷却时间（秒）
```

> 可以使用 `qsss config` 查看当前生效的关键参数，使用 `qsss sources` 与 `scripts/analyze_logs.py` 联合排查数据源的降级 / 恢复情况。

## 性能优化

### 性能提升
- **数据加载**: 3-5倍提升（通过多级缓存）
- **并发处理**: 4-8倍提升（通过并行计算）
- **数据库查询**: 5-10倍提升（通过索引优化）
- **缓存命中率**: 80%+（通过智能缓存策略）

### 性能测试
```bash
# 运行性能测试
python scripts/performance_test.py

# 查看性能报告
cat performance_test_results.json
```

## 日志分析工具

项目提供了一个专门的日志分析脚本，用于统计数据源降级 / 恢复事件：

```bash
python scripts/analyze_logs.py --log-file logs/qsss.log --group-by day
```

常用示例：

```bash
# 按天统计所有数据源的降级 / 恢复次数（使用默认日志路径设置）
python scripts/analyze_logs.py

# 指定日志文件并按小时聚合
python scripts/analyze_logs.py --log-file /path/to/qsss.log --group-by hour

# 只看 pytdx 和 tushare 两个数据源
python scripts/analyze_logs.py --source pytdx --source tushare

# 只统计 2025-01-01 之后的事件
python scripts/analyze_logs.py --since "2025-01-01"

# 指定精确时间区间
python scripts/analyze_logs.py --since "2025-01-01 09:30:00" --until "2025-01-02 23:59:59"
```

脚本会输出每个数据源在各个时间桶（全部 / 按天 / 按小时）的降级和恢复次数统计，
便于排查某个数据源是否经常降级、是否能及时恢复等稳定性问题。

## Docker部署

```bash
# 在项目根目录构建镜像
# （镜像内会安装核心引擎和 web 端依赖，并默认监听 8080 端口）
docker build -t qsss:latest .

# 以默认配置启动容器（生产环境，端口 8080）
docker run -d \
  --name qsss-app \
  -p 8080:8080 \
  -e DATABASE_URL=sqlite:///qsss_prod.db \
  qsss:latest

# 查看日志
docker logs -f qsss-app

# 访问：http://localhost:8080

# 如需自定义数据目录或 Redis 等，可额外挂载/配置：
# -v $(pwd)/data:/app/data \
# -e REDIS_URL=redis://redis:6379/0 \
```

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 技术支持

- **项目地址**: https://github.com/MisonL/QSSS
- **问题反馈**: https://github.com/MisonL/QSSS/issues
- **联系邮箱**: 1360962086@qq.com

## 免责声明

**重要声明**:
- 本系统基于历史数据和技术分析，不构成投资建议
- 任何投资决策需结合市场环境和个人风险承受能力
- 股市有风险，投资需谨慎
- 回测结果不代表未来收益表现

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

---
**⭐ 如果这个项目对您有帮助，请给个Star支持一下！**
