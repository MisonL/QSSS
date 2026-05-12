"""Package import contract tests."""

import os
import subprocess
import sys
from pathlib import Path


def test_import_qsss_does_not_import_lightgbm():
    """Importing package metadata should not require optional ML runtime."""
    sys.modules.pop("qsss", None)
    sys.modules.pop("lightgbm", None)

    import qsss

    assert qsss.__version__ == "2.0.0"
    assert "lightgbm" not in sys.modules


def _assert_clean_import_does_not_load_lightgbm(statement):
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; {statement}; print('lightgbm' in sys.modules)",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_import_quant_strategy_does_not_import_lightgbm():
    """Filtering-only strategy code should not require optional ML runtime."""
    _assert_clean_import_does_not_load_lightgbm(
        "from qsss.core.strategy import QuantStrategy; QuantStrategy()"
    )


def test_import_optimized_strategy_does_not_import_lightgbm():
    """Optimized strategy import should not load LightGBM until ML analysis."""
    _assert_clean_import_does_not_load_lightgbm(
        "from qsss.core.optimized_strategy import OptimizedQuantStrategy"
    )
