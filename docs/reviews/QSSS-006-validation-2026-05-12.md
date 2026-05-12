# QSSS-006 Validation - 2026-05-12

## Task

- ID: QSSS-006
- Title: 设计 SQLite MVP 落库
- Scope: 最小 SQLite schema 与可重复初始化脚本。

## Tables

- `stocks`
- `bars_daily`
- `quotes_snapshot`
- `concept_boards`
- `board_members`
- `board_flow_snapshots`
- `watchlists`
- `signals`

## Verification

Command:

```bash
uv run --with pytest --with pandas --with numpy --with pytdx --with pydantic --with pydantic-settings --with loguru --with click --with scikit-learn --with lightgbm pytest -q tests/test_sqlite_schema_contract.py
```

Result:

- Exit code: 0
- Summary: 3 passed

Command:

```bash
tmpdb=$(mktemp -t qsss_schema_XXXXXX.db); uv run --with pandas python scripts/init_sqlite.py "$tmpdb"; uv run --with pandas python scripts/init_sqlite.py "$tmpdb"; rm -f "$tmpdb"
```

Result:

- Exit code: 0
- Summary: initializer ran twice against the same SQLite path.

## Review Notes

- Core market keys use text dates and codes, and price/volume/amount fields use SQLite `REAL`.
- Duplicate daily bar rows are rejected by `PRIMARY KEY (symbol, date)`.
- This task creates schema only; it does not add background ingestion or live write paths.
