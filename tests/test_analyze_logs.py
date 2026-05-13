"""scripts/analyze_logs 日志分析工具的单元测试。

这些测试验证：
- 单行日志中降级 / 恢复事件的解析是否正确；
- analyze_log_file 在按天 / 按小时分组时的聚合结果；
- since / until 时间过滤与 source 过滤是否按预期工作。
"""

import textwrap
from pathlib import Path

import pytest

from qsss.scripts.analyze_logs import (
    _parse_filter_datetime,
    analyze_log_file,
    parse_event_from_line,
)


def _write_sample_log(tmp_path: Path) -> Path:
    """写入一个包含多数据源、多时间点事件的示例日志文件。"""

    content = textwrap.dedent(
        """
        2025-01-01 09:00:00 | INFO | 数据源 pytdx 连续失败 ... 进入降级状态 ...
        2025-01-01 10:00:00 | INFO | 数据源 pytdx 已恢复正常 ...
        2025-01-01 11:30:00 | INFO | 数据源 tushare 连续失败 ... 进入降级状态 ...
        2025-01-02 09:15:00 | INFO | 数据源 pytdx 连续失败 ... 进入降级状态 ...
        """
    ).strip()

    log_file = tmp_path / "qsss_sample.log"
    log_file.write_text(content + "\n", encoding="utf-8")
    return log_file


def test_parse_event_from_line_degrade_and_recover():
    """基础降级 / 恢复行能被正确解析。"""

    degrade_line = (
        "2025-01-01 10:00:00.123 | INFO | 数据源 pytdx 连续失败 ... 进入降级状态 ..."
    )
    recover_line = "2025-01-01 11:00:00 | INFO | 数据源 pytdx 已恢复正常 ..."

    event1 = parse_event_from_line(degrade_line)
    assert event1 is not None
    assert event1.source == "pytdx"
    assert event1.event_type == "degraded"
    assert event1.timestamp is not None
    assert event1.timestamp.year == 2025
    assert event1.timestamp.hour == 10

    event2 = parse_event_from_line(recover_line)
    assert event2 is not None
    assert event2.source == "pytdx"
    assert event2.event_type == "recovered"

    # 不相关的行应返回 None
    assert parse_event_from_line("some random line") is None


def test_analyze_log_file_group_by_day(tmp_path):
    """按天分组时，每个数据源 / 日期的事件计数应正确。"""

    log_file = _write_sample_log(tmp_path)

    stats = analyze_log_file(log_path=log_file, group_by="day")

    # 有两个数据源
    assert set(stats.keys()) == {"pytdx", "tushare"}

    # pytdx 在 2025-01-01：1 次降级 + 1 次恢复
    bucket_0101 = stats["pytdx"]["2025-01-01"]
    assert bucket_0101["degraded"] == 1
    assert bucket_0101["recovered"] == 1

    # pytdx 在 2025-01-02：1 次降级
    bucket_0102 = stats["pytdx"]["2025-01-02"]
    assert bucket_0102["degraded"] == 1
    assert bucket_0102["recovered"] == 0

    # tushare 只有 2025-01-01 的一次降级
    t_bucket = stats["tushare"]["2025-01-01"]
    assert t_bucket["degraded"] == 1
    assert t_bucket["recovered"] == 0


def test_analyze_log_file_group_by_hour(tmp_path):
    """按小时分组时，时间桶标签应符合预期格式。"""

    log_file = _write_sample_log(tmp_path)

    stats = analyze_log_file(log_path=log_file, group_by="hour")

    pytdx_buckets = stats["pytdx"]
    assert "2025-01-01 09:00" in pytdx_buckets
    assert "2025-01-01 10:00" in pytdx_buckets
    assert "2025-01-02 09:00" in pytdx_buckets


def test_analyze_log_file_since_until_and_source_filter(tmp_path):
    """时间过滤和数据源过滤应共同生效。"""

    log_file = _write_sample_log(tmp_path)

    # 过滤掉 09:00 的降级事件，只保留 10:00 之后的 pytdx 事件
    since = _parse_filter_datetime("2025-01-01 09:30:00")
    until = _parse_filter_datetime("2025-01-02 23:59:59")

    stats = analyze_log_file(
        log_path=log_file,
        group_by="day",
        sources_filter=["pytdx"],
        since=since,
        until=until,
    )

    # 只包含 pytdx
    assert set(stats.keys()) == {"pytdx"}

    # 2025-01-01 只保留恢复事件
    bucket_0101 = stats["pytdx"]["2025-01-01"]
    assert bucket_0101["degraded"] == 0
    assert bucket_0101["recovered"] == 1

    # 2025-01-02 只保留一次降级
    bucket_0102 = stats["pytdx"]["2025-01-02"]
    assert bucket_0102["degraded"] == 1
    assert bucket_0102["recovered"] == 0


def test_parse_filter_datetime_invalid_format():
    """非法时间格式应抛出 ValueError。"""

    with pytest.raises(ValueError):
        _parse_filter_datetime("2025/01/01")
