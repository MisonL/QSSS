"""Dependency declaration contract tests."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREE_DATA_SOURCE_PACKAGES = {"pytdx", "akshare", "baostock"}


def _requirement_name(requirement: str) -> str:
    line = requirement.strip()
    if not line or line.startswith("#"):
        return ""
    match = re.match(r"([A-Za-z0-9_.-]+)", line)
    return match.group(1).lower().replace("_", "-") if match else ""


def _requirements_names(path: Path) -> set[str]:
    return {
        name
        for name in (_requirement_name(line) for line in path.read_text().splitlines())
        if name
    }


def _project_dependency_names() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return {_requirement_name(dep) for dep in data["project"]["dependencies"]}


def test_pyproject_declares_free_data_source_runtime_dependencies():
    """Free-first data source packages should be first-class runtime deps."""
    project_deps = _project_dependency_names()

    assert FREE_DATA_SOURCE_PACKAGES <= project_deps


def test_requirements_txt_matches_pyproject_runtime_dependency_names():
    """Root requirements.txt should mirror pyproject runtime dependency names."""
    project_deps = _project_dependency_names()
    root_requirements = _requirements_names(REPO_ROOT / "requirements.txt")

    assert root_requirements == project_deps


def test_web_requirements_include_all_pyproject_runtime_dependencies():
    """Web requirements should include core runtime deps plus web-only deps."""
    project_deps = _project_dependency_names()
    web_requirements = _requirements_names(REPO_ROOT / "web" / "requirements.txt")

    assert project_deps <= web_requirements
