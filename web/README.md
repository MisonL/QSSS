# QSSS Web Application

QSSS量化选股系统的Web界面，基于Flask框架开发。

## 功能特性

- **股票分析**: 多因子模型和机器学习技术分析
- **回测功能（规划中）**: 已预留数据模型与接口，占位但当前版本未提供实际回测逻辑
- **可视化图表**: ECharts集成，支持K线图和回测结果展示
- **异步处理**: Celery + Redis异步任务队列
- **响应式设计**: Bootstrap 5移动端适配
- **配置管理**: 支持多环境配置

## 快速开始

### 1. 安装依赖

```bash
cd web
pip install -r requirements.txt
```

### 2. 启动Redis (用于Celery)

```bash
# macOS
brew install redis
redis-server

# Linux
sudo apt-get install redis-server
redis-server
```

### 3. 初始化数据库

```bash
python run.py --init-db
```

### 4. 启动应用

#### 开发环境
```bash
python run.py --env dev --host 0.0.0.0 --port 5000
```

#### 生产环境
```bash
python run.py --env prod --host 0.0.0.0 --port 8080
```

### 5. 启动Celery Worker (用于异步任务)

```bash
celery -A app.celery worker --loglevel=info
```

## 项目结构

```
web/
+-- app.py              # Flask应用主文件
+-- models.py           # 数据库模型
+-- routes.py           # 路由和视图函数
+-- tasks.py            # Celery异步任务
+-- config.py           # 配置文件
+-- run.py              # 启动脚本
+-- requirements.txt    # 依赖列表
+-- static/             # 静态文件
|   +-- css/
|   |   +-- style.css   # 自定义样式
|   +-- js/
|       +-- main.js     # JavaScript工具函数
+-- templates/          # HTML模板
    +-- base.html       # 基础模板
    +-- index.html      # 首页
    +-- analysis.html   # 分析页面
    +-- backtest.html   # 回测页面
```

## API接口

### 股票分析
- `POST /api/analyze` - 提交股票分析任务
- `GET /api/task/<task_id>` - 获取任务状态

### 策略回测（规划中）
- `POST /api/backtest` - 当前返回 501，提示“回测功能尚未实现”，用于保留接口占位

## 环境变量

```bash
# 数据库
DATABASE_URL=sqlite:///qsss_web.db

# Redis
REDIS_URL=redis://localhost:6379/0

# 密钥
SECRET_KEY=your-secret-key-here

# 日志级别
LOG_LEVEL=INFO
```

## 部署说明

### 使用Docker (推荐)

```bash
# 构建镜像
docker build -t qsss-web .

# 运行容器
docker run -d \
  -p 8080:8080 \
  -e DATABASE_URL=sqlite:///data/qsss.db \
  -e REDIS_URL=redis://redis:6379/0 \
  --name qsss-web \
  qsss-web
```

### 使用Waitress (WSGI)

```bash
# 生产环境启动
python run.py --env prod
```

## 开发指南

### 添加新策略

1. 在 `src/qsss/strategies/` 中创建策略文件
2. 在 `web/models.py` 中添加策略配置
3. 更新 `web/routes.py` 中的API接口

### 自定义图表

使用ECharts创建自定义图表：

```javascript
// 创建K线图
const chart = QSSS.createKLineChart('chart-container', {
    dates: ['2024-01-01', '2024-01-02', ...],
    values: [[100, 105, 98, 103], [103, 108, 102, 107], ...],
    ma5: [101, 103, ...],
    ma10: [102, 104, ...],
    ma20: [103, 105, ...]
});
```

## 故障排除

### 常见问题

1. **Redis连接失败**
   - 确保Redis服务已启动
   - 检查 `REDIS_URL` 配置

2. **数据库初始化失败**
   - 运行 `python run.py --init-db`
   - 检查数据库文件权限

3. **Celery任务不执行**
   - 确保已启动Celery worker
   - 检查Redis连接

### 日志查看

```bash
# 查看应用日志
tail -f web/logs/app.log

# 查看Celery日志
celery -A app.celery worker --loglevel=debug
```

## 许可证

MIT License