from celery import Celery
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from web.config import apply_config

app = Flask(__name__)

# Configuration
ACTIVE_CONFIG_NAME = apply_config(app)

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"])
celery.conf.update(app.config)

"""核心 Flask 应用对象模块。

不在此模块内导入 models/routes，以避免初始化时的循环依赖；
由启动脚本（如 web/run.py）负责在应用启动前导入 ``web.models``
和 ``web.routes`` 完成路由与模型注册。
"""
