"""Tonghuashun local board cache reader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


@dataclass(frozen=True)
class BoardMember:
    board_code: str
    board_name: str
    board_type: str
    symbol: str
    name: str
    source: str
    version: str


class ThsLocalBoardAdapter:
    """Read only parser for Tonghuashun local board membership ini files."""

    def __init__(
        self,
        conception_path: Optional[str | Path] = None,
        industry_path: Optional[str | Path] = None,
    ) -> None:
        self.paths: Dict[str, Optional[Path]] = {
            "concept": Path(conception_path) if conception_path else None,
            "industry": Path(industry_path) if industry_path else None,
        }

    def get_board_members(
        self, board: str, board_type: str = "concept"
    ) -> pd.DataFrame:
        """Return members for a board name or generated board code."""
        board_type = _normalize_board_type(board_type)
        path = self._require_path(board_type)
        boards = _parse_board_file(path, board_type)
        board_key = str(board).strip()
        for board_name, members in boards.items():
            board_code = _stable_board_code(board_type, board_name)
            if board_key in {board_name, board_code}:
                return _members_to_frame(members)
        return _members_to_frame([])

    def get_all_board_members(self, board_type: str = "concept") -> pd.DataFrame:
        """Return members for all boards in a local cache file."""
        board_type = _normalize_board_type(board_type)
        path = self._require_path(board_type)
        boards = _parse_board_file(path, board_type)
        members = [member for group in boards.values() for member in group]
        return _members_to_frame(members)

    def _require_path(self, board_type: str) -> Path:
        path = self.paths.get(board_type)
        if path is None:
            filename = _default_filename(board_type)
            raise FileNotFoundError(f"未配置同花顺本地缓存文件: {filename}")
        if not path.exists():
            filename = _default_filename(board_type)
            raise FileNotFoundError(f"同花顺本地缓存文件不存在: {filename} ({path})")
        if not path.is_file():
            raise FileNotFoundError(f"同花顺本地缓存路径不是文件: {path}")
        return path


def _parse_board_file(path: Path, board_type: str) -> Dict[str, list[BoardMember]]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    version = _file_version(path)
    boards: Dict[str, list[BoardMember]] = {}
    current_board = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        section = _parse_section(line)
        if section:
            current_board = section
            boards.setdefault(current_board, [])
            continue
        if not current_board:
            continue
        boards[current_board].extend(
            _parse_member_line(line, current_board, board_type, version)
        )
    return boards


def _parse_section(line: str) -> str:
    if line.startswith("[") and line.endswith("]"):
        return line[1:-1].strip()
    return ""


def _parse_member_line(
    line: str, board_name: str, board_type: str, version: str
) -> list[BoardMember]:
    members = []
    for symbol in CODE_RE.findall(line):
        members.append(
            BoardMember(
                board_code=_stable_board_code(board_type, board_name),
                board_name=board_name,
                board_type=board_type,
                symbol=symbol,
                name=_extract_name(line, symbol),
                source="ths_local_cache",
                version=version,
            )
        )
    return members


def _extract_name(line: str, symbol: str) -> str:
    key, sep, value = line.partition("=")
    if sep and key.strip() == symbol:
        return value.strip()
    for splitter in (":", "|"):
        left, marker, right = line.partition(splitter)
        if marker and symbol in {left.strip(), right.strip()}:
            return right.strip() if left.strip() == symbol else left.strip()
    return ""


def _members_to_frame(members: Iterable[BoardMember]) -> pd.DataFrame:
    columns = [
        "board_code",
        "board_name",
        "board_type",
        "symbol",
        "name",
        "source",
        "version",
    ]
    rows = [member.__dict__ for member in members]
    return pd.DataFrame(rows, columns=columns)


def _stable_board_code(board_type: str, board_name: str) -> str:
    digest = sha1(f"{board_type}:{board_name}".encode("utf-8")).hexdigest()
    return f"THS{digest[:10].upper()}"


def _file_version(path: Path) -> str:
    modified_at = datetime.fromtimestamp(path.stat().st_mtime)
    return f"{path.name}:{modified_at.isoformat(timespec='seconds')}"


def _normalize_board_type(board_type: str) -> str:
    normalized = str(board_type).strip().lower()
    if normalized not in {"concept", "industry"}:
        raise ValueError(f"不支持的板块类型: {board_type}")
    return normalized


def _default_filename(board_type: str) -> str:
    if board_type == "concept":
        return "block_conception.ini"
    return "block_industry.ini"
