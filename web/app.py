import os

from celery import Celery
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configuration
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "dev-secret-key-change-in-production"
)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///qsss_web.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["CELERY_BROKER_URL"] = os.environ.get(
    "REDIS_URL", "redis://localhost:6379/0"
)
app.config["CELERY_RESULT_BACKEND"] = os.environ.get(
    "REDIS_URL", "redis://localhost:6379/0"
)

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

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
