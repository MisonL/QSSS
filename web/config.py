import os
import secrets
import threading
from pathlib import Path
from typing import Any

PRODUCTION_ENV_VALUES = {"production", "prod"}
CONFIG_ALIASES = {
    "dev": "development",
    "development": "development",
    "test": "testing",
    "testing": "testing",
    "prod": "production",
    "production": "production",
}
PRODUCTION_REQUIRED_CONFIG = {
    "SECRET_KEY": "SECRET_KEY",
    "SQLALCHEMY_DATABASE_URI": "DATABASE_URL",
    "CELERY_BROKER_URL": "REDIS_URL",
    "CELERY_RESULT_BACKEND": "REDIS_URL",
}
_secret_key_lock = threading.Lock()


def is_production_environment() -> bool:
    """Return whether the process is explicitly configured as production."""
    env_name = _configured_environment_name()
    return env_name.strip().lower() in PRODUCTION_ENV_VALUES


def _configured_environment_name() -> str:
    return (
        os.environ.get("QSSS_WEB_ENV")
        or os.environ.get("FLASK_ENV")
        or os.environ.get("APP_ENV")
        or ""
    )


def load_secret_key(production: bool | None = None) -> str:
    """Load Flask SECRET_KEY from env or a stable key file."""
    secret_key = os.environ.get("SECRET_KEY")
    if secret_key:
        return secret_key
    secret_key_file = _secret_key_file_path()
    is_production = is_production_environment() if production is None else production
    with _secret_key_lock:
        if secret_key_file.exists():
            stored_secret = secret_key_file.read_text(encoding="utf-8").strip()
            if stored_secret:
                return stored_secret
        if is_production:
            raise RuntimeError("生产环境缺少必要配置: SECRET_KEY")
        generated_secret = secrets.token_hex(32)
        secret_key_file.parent.mkdir(parents=True, exist_ok=True)
        secret_key_file.write_text(generated_secret, encoding="utf-8")
        secret_key_file.chmod(0o600)
        return generated_secret


def _secret_key_file_path() -> Path:
    configured = os.environ.get("SECRET_KEY_FILE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "instance" / "secret_key"


def resolve_config_name(config_name: str | None = None) -> str:
    """Resolve config aliases from CLI or process environment."""
    raw_name = config_name or _configured_environment_name() or "development"
    resolved = CONFIG_ALIASES.get(raw_name.strip().lower())
    if not resolved:
        raise RuntimeError(f"不支持的 Web 配置环境: {raw_name}")
    return resolved


def validate_runtime_config(app_config: Any, config_name: str) -> None:
    """Check production configuration that must come from external env."""
    if config_name != "production":
        return
    missing_env_names = {
        env_name
        for config_key, env_name in PRODUCTION_REQUIRED_CONFIG.items()
        if not app_config.get(config_key)
    }
    if missing_env_names:
        missing = ", ".join(sorted(missing_env_names))
        raise RuntimeError(f"生产环境缺少必要配置: {missing}")


def apply_config(flask_app: Any, config_name: str | None = None) -> str:
    """Apply the selected Flask config and validate runtime requirements."""
    resolved_name = resolve_config_name(config_name)
    flask_app.config.from_object(config[resolved_name])
    flask_app.config["SECRET_KEY"] = load_secret_key(
        production=resolved_name == "production"
    )
    validate_runtime_config(flask_app.config, resolved_name)
    return resolved_name


class Config:
    """基础配置"""

    SECRET_KEY = None

    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or "sqlite:///qsss_web.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Celery配置
    CELERY_BROKER_URL = os.environ.get("REDIS_URL") or "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND = os.environ.get("REDIS_URL") or "redis://localhost:6379/0"

    # 分页配置
    POSTS_PER_PAGE = 20

    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # 缓存配置
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300

    # 日志配置
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "app.log")


class DevelopmentConfig(Config):
    """开发环境配置"""

    DEBUG = True
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DEV_DATABASE_URL") or "sqlite:///qsss_dev.db"
    )


class TestingConfig(Config):
    """测试环境配置"""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("TEST_DATABASE_URL") or "sqlite:///:memory:"
    )
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """生产环境配置"""

    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    CELERY_BROKER_URL = os.environ.get("REDIS_URL")
    CELERY_RESULT_BACKEND = os.environ.get("REDIS_URL")

    # 生产环境安全配置
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
