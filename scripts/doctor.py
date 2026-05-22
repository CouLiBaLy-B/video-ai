"""Local environment doctor for Video AI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

CHECKS = {
    "fastapi": "fastapi",
    "pydantic": "pydantic",
    "httpx": "httpx",
    "pillow": "PIL",
    "sqlalchemy_optional": "sqlalchemy",
    "deepagents_optional": "deepagents",
    "diffusers_optional": "diffusers",
    "redis_optional": "redis",
    "rq_optional": "rq",
}


def module_available(module: str) -> bool:
    """Return whether a module can be imported."""
    return importlib.util.find_spec(module) is not None


def main() -> int:
    """Print environment diagnostics as JSON."""
    result = {
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "modules": {name: module_available(module) for name, module in CHECKS.items()},
        "workspace": {
            "api": Path("apps/api/src/video_ai").exists(),
            "web": Path("apps/web/package.json").exists(),
            "skills": Path("skills").exists(),
            "migrations": Path("migrations").exists(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    required_modules = ["fastapi", "pydantic", "httpx", "pillow"]
    required_ok = all(result["modules"][name] for name in required_modules)
    workspace_ok = all(result["workspace"].values())
    return 0 if required_ok and workspace_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
