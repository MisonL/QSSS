#!/usr/bin/env python3
"""QSSS 日志分析工具（包内版本）

该模块与项目根目录下的 ``scripts/analyze_logs.py`` 提供相同功能，
便于在测试和库代码中通过 ``src.qsss.scripts.analyze_logs`` 进行导入。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 日志行解析
# ---------------------------------------------------------------------------

# 尝试从行首解析时间戳，兼容 loguru 默认格式：
# 2025-01-01 12:34:56.789 | INFO     | ...
TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)"
)

# 降级 / 恢复事件的消息模式（与 DataManager 中的日志保持一致）
DEGRADE_RE = re.compile(r"数据源\s+(?P<source>\w+)\s+连续失败")
RECOVER_RE = re.compile(r"数据源\s+(?P<source>\w+)\s+已恢复正常")


@dataclass
class Event:
    """单条降级 / 恢复事件。"""

    timestamp: Optional[datetime]
    source: str
    event_type: str  # "degraded" or "recovered"
    raw_line: str


def _parse_timestamp_from_prefix(prefix: str) -> Optional[datetime]:
    """从日志行前缀中解析时间戳。

    优先尝试常见的几种格式；如果全部失败，则返回 None。
    """

    # 将 `T` 统一替换为空格，方便 strptime
    normalized = prefix.replace("T", " ", 1)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def parse_event_from_line(line: str) -> Optional[Event]:
    """从单行日志中解析降级 / 恢复事件。

    只关心包含：
    - "进入降级状态"  （连续失败达到阈值）
    - "已恢复正常"    （从降级状态恢复）
    """

    line = line.rstrip("\n")

    # 时间戳（如果存在）
    ts_match = TIMESTAMP_RE.match(line)
    timestamp: Optional[datetime] = None
    if ts_match:
        timestamp = _parse_timestamp_from_prefix(ts_match.group("ts"))

    if "进入降级状态" in line:
        m = DEGRADE_RE.search(line)
        if not m:
            return None
        return Event(
            timestamp=timestamp,
            source=m.group("source"),
            event_type="degraded",
            raw_line=line,
        )

    if "已恢复正常" in line:
        m = RECOVER_RE.search(line)
        if not m:
            return None
        return Event(
            timestamp=timestamp,
            source=m.group("source"),
            event_type="recovered",
            raw_line=line,
        )

    return None


# ---------------------------------------------------------------------------
# 统计与聚合
# ---------------------------------------------------------------------------


def _parse_filter_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"无法解析时间参数: {value!r}，请使用 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS' 格式"
    )


def _group_key(ts: Optional[datetime], group_by: str) -> str:
    if not ts or group_by == "none":
        return "ALL"
    if group_by == "day":
        return ts.strftime("%Y-%m-%d")
    if group_by == "hour":
        return ts.strftime("%Y-%m-%d %H:00")
    # 理论上不会到这里
    return "ALL"


def analyze_log_file(
    log_path: Path,
    *,
    group_by: str = "day",
    sources_filter: Optional[List[str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    encoding: str = "utf-8",
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """解析日志文件并返回聚合统计结果。

    返回结构： stats[source][bucket][event_type] = count
    其中 bucket 由 group_by 决定：
      - "none" -> "ALL"
      - "day"  -> "YYYY-MM-DD"
      - "hour" -> "YYYY-MM-DD HH:00"
    """

    stats: Dict[str, Dict[str, Dict[str, int]]] = {}

    if sources_filter:
        sources_filter = [s.strip() for s in sources_filter if s.strip()]

    with log_path.open("r", encoding=encoding, errors="ignore") as f:
        for line in f:
            event = parse_event_from_line(line)
            if not event:
                continue

            # 过滤数据源
            if sources_filter and event.source not in sources_filter:
                continue

            # 时间范围过滤：无法解析时间戳的行在指定 since/until 时会被跳过
            if (since or until) and event.timestamp is None:
                continue
            if since and event.timestamp and event.timestamp < since:
                continue
            if until and event.timestamp and event.timestamp > until:
                continue

            bucket = _group_key(event.timestamp, group_by)
            source_stats = stats.setdefault(event.source, {})
            bucket_stats = source_stats.setdefault(
                bucket, {"degraded": 0, "recovered": 0}
            )
            bucket_stats[event.event_type] += 1

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _guess_default_log_file() -> Path:
    """根据约定猜测默认日志文件路径。

    优先：
      1. settings.log_file（如果可以成功导入）
      2. 项目相对路径 logs/qsss.log
    """

    # 延迟导入，避免在脚本单独使用时强依赖项目结构
    candidates: List[Path] = []
    try:
        from src.qsss.config.settings import settings as qsss_settings  # type: ignore

        log_file = getattr(qsss_settings, "log_file", None)
        if isinstance(log_file, str) and log_file:
            candidates.append(Path(log_file))
    except Exception:  # pragma: no cover - 配置导入失败时退化为默认路径
        pass

    candidates.append(Path("logs/qsss.log"))

    # 返回第一个已存在的文件，否则返回第一个候选路径
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="QSSS 日志分析工具 - 统计数据源降级 / 恢复事件次数",
    )
    parser.add_argument(
        "--log-file",
        "-f",
        type=str,
        default=None,
        help="日志文件路径（默认尝试使用 settings.log_file 或 logs/qsss.log）",
    )
    parser.add_argument(
        "--group-by",
        choices=["none", "day", "hour"],
        default="day",
        help="聚合粒度：none=不分组, day=按日, hour=按小时",
    )
    parser.add_argument(
        "--source",
        "-s",
        action="append",
        help="只统计指定数据源（例如: pytdx, tushare, baostock），可多次使用",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="只统计此时间之后的事件 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="只统计此时间之前的事件 (YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="日志文件编码，默认 utf-8",
    )

    args = parser.parse_args(argv)

    log_path = Path(args.log_file) if args.log_file else _guess_default_log_file()
    if not log_path.exists():
        raise SystemExit(f"日志文件不存在: {log_path}")

    try:
        since_dt = _parse_filter_datetime(args.since)
        until_dt = _parse_filter_datetime(args.until)
    except ValueError as e:
        raise SystemExit(str(e))

    stats = analyze_log_file(
        log_path=log_path,
        group_by=args.group_by,
        sources_filter=args.source,
        since=since_dt,
        until=until_dt,
        encoding=args.encoding,
    )

    if not stats:
        print(f"在日志文件 {log_path} 中未找到任何降级/恢复事件。")
        return

    print(f"日志文件: {log_path}")
    if since_dt or until_dt:
        print("时间范围:")
        if since_dt:
            print(f"  自: {since_dt}")
        if until_dt:
            print(f"  至: {until_dt}")
    if args.group_by != "none":
        print(f"聚合粒度: {args.group_by}")
    if args.source:
        print(f"过滤数据源: {', '.join(args.source)}")
    print("")

    # 按数据源名称排序输出
    for source in sorted(stats.keys()):
        buckets = stats[source]
        # 统计总数
        total_degraded = sum(b["degraded"] for b in buckets.values())
        total_recovered = sum(b["recovered"] for b in buckets.values())

        print(f"=== 数据源: {source} ===")
        print(f"总降级次数: {total_degraded}")
        print(f"总恢复次数: {total_recovered}")

        # 按时间分桶输出
        for bucket in sorted(buckets.keys()):
            if bucket == "ALL" and args.group_by == "none":
                label = "ALL"
            else:
                label = bucket
            degraded = buckets[bucket]["degraded"]
            recovered = buckets[bucket]["recovered"]
            print(f"  {label}: 降级={degraded}, 恢复={recovered}")
        print("")


if __name__ == "__main__":  # pragma: no cover
    main()
