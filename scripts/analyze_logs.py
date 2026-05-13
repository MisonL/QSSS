#!/usr/bin/env python3
"""QSSS 日志分析工具（脚本入口）

实际实现位于 ``src.qsss.scripts.analyze_logs`` 中，这里仅作为
在项目根目录下直接运行脚本的兼容入口，例如：

    python scripts/analyze_logs.py --log-file logs/qsss.log --group-by day
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qsss.scripts.analyze_logs import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    main()
