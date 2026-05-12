# QSSS Web Application Dockerfile
# 基于官方 Python 镜像，提供运行 Web 界面的基础环境

FROM python:3.10-slim

WORKDIR /app

# 安装系统级运行时依赖（LightGBM 等需要 libgomp）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装运行时依赖（尽量使用二进制轮子，避免编译）
ENV PIP_NO_CACHE_DIR=1

# 仅复制构建所需文件，提升缓存命中率
COPY pyproject.toml README.md ./
COPY src ./src
COPY web ./web
COPY scripts ./scripts

# 安装核心引擎与 Web 端依赖
RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && python -m pip install -r web/requirements.txt

# 默认环境变量（可在 docker run 时覆盖）
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    QSSS_PRIMARY_DATA_SOURCE=pytdx

# 暴露 Web 端口
EXPOSE 8080

# 启动 Flask Web 应用（生产风格，监听 0.0.0.0:8080）
CMD ["python", "web/run.py", "--env", "prod", "--host", "0.0.0.0", "--port", "8080"]
