"""Initialize QSSS SQLite MVP schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from qsss.database.schema import initialize_sqlite


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize QSSS SQLite schema")
    parser.add_argument("db_path", type=Path, help="SQLite database path")
    args = parser.parse_args()
    path = initialize_sqlite(args.db_path)
    print(f"SQLite schema initialized: {path}")


if __name__ == "__main__":
    main()
