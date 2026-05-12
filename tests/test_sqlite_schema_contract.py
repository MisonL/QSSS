"""SQLite MVP schema contract tests."""

import sqlite3

from qsss.database.schema import initialize_sqlite


EXPECTED_TABLES = {
    "stocks",
    "bars_daily",
    "quotes_snapshot",
    "concept_boards",
    "board_members",
    "board_flow_snapshots",
    "watchlists",
    "signals",
}


def _table_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1]: row[2].upper() for row in rows}


def test_sqlite_schema_initializes_expected_tables(tmp_path):
    """SQLite initializer should create the MVP market data tables."""
    db_path = tmp_path / "qsss.db"

    initialize_sqlite(db_path)
    initialize_sqlite(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert EXPECTED_TABLES.issubset(tables)


def test_sqlite_schema_locks_key_column_types(tmp_path):
    """Critical SQLite columns should keep stable storage types."""
    db_path = tmp_path / "qsss.db"
    initialize_sqlite(db_path)

    with sqlite3.connect(db_path) as conn:
        assert _table_columns(conn, "stocks")["symbol"] == "TEXT"
        assert _table_columns(conn, "bars_daily")["date"] == "TEXT"
        assert _table_columns(conn, "quotes_snapshot")["price"] == "REAL"
        assert _table_columns(conn, "board_members")["board_code"] == "TEXT"
        assert _table_columns(conn, "signals")["payload_json"] == "TEXT"


def test_sqlite_schema_enforces_unique_market_keys(tmp_path):
    """SQLite schema should reject duplicate core market rows."""
    db_path = tmp_path / "qsss.db"
    initialize_sqlite(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO stocks(symbol, name, market) VALUES (?, ?, ?)",
            ("000001", "平安银行", "深交所-主板"),
        )
        conn.execute(
            """
            INSERT INTO bars_daily(symbol, date, open, close, high, low, volume, amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("000001", "2026-05-12", 10, 11, 12, 9, 1000, 10000),
        )

        try:
            conn.execute(
                """
                INSERT INTO bars_daily(
                    symbol, date, open, close, high, low, volume, amount
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("000001", "2026-05-12", 10, 11, 12, 9, 1000, 10000),
            )
        except sqlite3.IntegrityError:
            duplicate_rejected = True
        else:
            duplicate_rejected = False

    assert duplicate_rejected
