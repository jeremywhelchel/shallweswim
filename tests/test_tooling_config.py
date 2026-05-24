import json
import tomllib
from pathlib import Path


def test_python_and_frontend_playwright_versions_stay_in_sync() -> None:
    """Python and Node Playwright share browser cache only when versions match."""
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    package_json = json.loads((root / "frontend" / "package.json").read_text())

    python_spec = next(
        dep
        for dep in pyproject["dependency-groups"]["dev"]
        if dep.startswith("playwright")
    )
    assert python_spec.startswith("playwright==")

    python_version = python_spec.removeprefix("playwright==")
    frontend_version = package_json["devDependencies"]["@playwright/test"]

    assert frontend_version == python_version
