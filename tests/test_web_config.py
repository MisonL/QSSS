"""Web configuration contract tests."""

import os
import stat
import subprocess
import sys
import threading
import importlib.util
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from web.config import is_production_environment, load_secret_key

WEB_DEPS_AVAILABLE = all(
    importlib.util.find_spec(package) is not None
    for package in ("celery", "flask", "flask_migrate", "flask_sqlalchemy")
)


def test_secret_key_missing_fails_in_explicit_production(monkeypatch, tmp_path):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("SECRET_KEY_FILE", str(tmp_path / "missing_secret_key"))
    monkeypatch.setenv("QSSS_WEB_ENV", "production")

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        load_secret_key()


def test_secret_key_can_be_persisted_outside_production(monkeypatch, tmp_path):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("QSSS_WEB_ENV", "development")
    secret_file = tmp_path / "secret_key"
    monkeypatch.setenv("SECRET_KEY_FILE", str(secret_file))

    first_key = load_secret_key()
    second_key = load_secret_key()

    assert len(first_key) == 64
    assert second_key == first_key
    assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600
    assert not is_production_environment()


def test_secret_key_file_can_supply_production_secret(monkeypatch, tmp_path):
    secret_file = tmp_path / "secret_key"
    secret_file.write_text("from-file-secret\n", encoding="utf-8")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("SECRET_KEY_FILE", str(secret_file))
    monkeypatch.setenv("QSSS_WEB_ENV", "production")

    assert load_secret_key() == "from-file-secret"


def test_web_app_import_fails_fast_when_production_dependencies_missing():
    if not WEB_DEPS_AVAILABLE:
        pytest.skip("Web 依赖未安装，跳过 web.app 导入契约测试")
    env = os.environ.copy()
    env["QSSS_WEB_ENV"] = "production"
    env["SECRET_KEY"] = "test-secret"
    env.pop("DATABASE_URL", None)
    env.pop("REDIS_URL", None)

    result = subprocess.run(
        [sys.executable, "-c", "import web.app"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "DATABASE_URL" in result.stderr
    assert "REDIS_URL" in result.stderr


def test_web_app_import_accepts_secret_key_file_in_production(tmp_path):
    if not WEB_DEPS_AVAILABLE:
        pytest.skip("Web 依赖未安装，跳过 web.app 导入契约测试")
    secret_file = tmp_path / "secret_key"
    secret_file.write_text("from-file-secret\n", encoding="utf-8")
    env = os.environ.copy()
    env["QSSS_WEB_ENV"] = "production"
    env["SECRET_KEY_FILE"] = str(secret_file)
    env["DATABASE_URL"] = "sqlite:///prod.db"
    env["REDIS_URL"] = "redis://localhost:6379/0"
    env.pop("SECRET_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from web.app import app; print(app.config['SECRET_KEY'])",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "from-file-secret"


def test_run_env_prod_fails_before_creating_development_secret(tmp_path):
    if not WEB_DEPS_AVAILABLE:
        pytest.skip("Web 依赖未安装，跳过 web.run 启动契约测试")
    secret_file = tmp_path / "secret_key"
    env = os.environ.copy()
    env.pop("QSSS_WEB_ENV", None)
    env.pop("FLASK_ENV", None)
    env.pop("APP_ENV", None)
    env.pop("SECRET_KEY", None)
    env["SECRET_KEY_FILE"] = str(secret_file)
    env["DATABASE_URL"] = "sqlite:///prod.db"
    env["REDIS_URL"] = "redis://localhost:6379/0"

    result = subprocess.run(
        [sys.executable, "web/run.py", "--env", "prod", "--init-db"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "SECRET_KEY" in result.stderr
    assert not secret_file.exists()


def test_development_secret_file_is_written_once_under_concurrency(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("QSSS_WEB_ENV", "development")
    secret_file = tmp_path / "secret_key"
    monkeypatch.setenv("SECRET_KEY_FILE", str(secret_file))
    results = []
    errors = []

    def load_key():
        try:
            results.append(load_secret_key())
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=load_key) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(set(results)) == 1
    assert secret_file.read_text(encoding="utf-8").strip() == results[0]
    assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600
