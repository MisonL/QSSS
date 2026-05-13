"""Pytest process defaults for deterministic local tests."""

import os

os.environ.setdefault("QSSS_WEB_ENV", "testing")
