# QSSS-007 Validation - 2026-05-12

## Task

- ID: QSSS-007
- Title: Web MVP 行情和板块页面
- Scope: `/market`、`/boards`、`/watchlists/ai-tech` 三个页面和导航入口。

## Verification

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm --with Flask --with Flask-SQLAlchemy --with Flask-Migrate --with celery pytest -q tests/test_web_mvp_pages.py
```

Result:

- Exit code: 0
- Summary: 1 passed, 1 warning

Browser smoke:

- Temporary Flask server: `http://127.0.0.1:5067`
- `/market`: rendered real pytdx quote rows for AI tech sample symbols.
- `/boards`: rendered AKShare concept board and fund flow tables. Eastmoney board endpoint failed once, then adapter used the explicit AKShare THS public board-list fallback.
- `/watchlists/ai-tech`: rendered real pytdx quote rows for the AI tech watchlist.

## Review Notes

- Pages do not contain static fake market rows; empty or failed data paths render explicit empty/error states.
- `/market` uses a bounded sample quote request instead of triggering full-market stock list loading.
