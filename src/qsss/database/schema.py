"""SQLite MVP schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    market TEXT,
    exchange TEXT,
    board TEXT,
    source TEXT NOT NULL DEFAULT 'unknown',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bars_daily (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    volume REAL NOT NULL,
    amount REAL NOT NULL,
    amplitude REAL,
    pct_chg REAL,
    change REAL,
    turn REAL,
    source TEXT NOT NULL DEFAULT 'unknown',
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
);

CREATE TABLE IF NOT EXISTS quotes_snapshot (
    symbol TEXT NOT NULL,
    quote_time TEXT NOT NULL,
    price REAL,
    last_close REAL,
    open REAL,
    high REAL,
    low REAL,
    volume REAL,
    amount REAL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, quote_time, source)
);

CREATE TABLE IF NOT EXISTS concept_boards (
    board_code TEXT PRIMARY KEY,
    board_name TEXT NOT NULL,
    board_type TEXT NOT NULL CHECK (board_type IN ('concept', 'industry')),
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (board_name, board_type, source)
);

CREATE TABLE IF NOT EXISTS board_members (
    board_code TEXT NOT NULL,
    board_name TEXT NOT NULL,
    board_type TEXT NOT NULL CHECK (board_type IN ('concept', 'industry')),
    symbol TEXT NOT NULL,
    name TEXT,
    source TEXT NOT NULL,
    version TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (board_code, symbol, source)
);

CREATE TABLE IF NOT EXISTS board_flow_snapshots (
    board_code TEXT NOT NULL,
    board_name TEXT NOT NULL,
    board_type TEXT NOT NULL CHECK (board_type IN ('concept', 'industry')),
    pct_chg REAL,
    amount REAL,
    net_inflow REAL,
    main_net_inflow REAL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (board_code, fetched_at, source)
);

CREATE TABLE IF NOT EXISTS watchlists (
    name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (name, symbol)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    signal_time TEXT NOT NULL,
    strategy TEXT NOT NULL,
    score REAL NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (symbol, signal_time, strategy)
);

CREATE INDEX IF NOT EXISTS idx_bars_daily_symbol_date
    ON bars_daily(symbol, date);
CREATE INDEX IF NOT EXISTS idx_quotes_snapshot_symbol_time
    ON quotes_snapshot(symbol, quote_time);
CREATE INDEX IF NOT EXISTS idx_board_members_symbol
    ON board_members(symbol);
CREATE INDEX IF NOT EXISTS idx_board_flow_type_time
    ON board_flow_snapshots(board_type, fetched_at);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time
    ON signals(symbol, signal_time);
"""


def initialize_sqlite(db_path: PathLike) -> Path:
    """Create or update the SQLite MVP schema idempotently."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    return path
