"""I/O helpers: panel loading, artifact save/load, run manifest."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PANEL_PATH, RUN_MANIFEST, SEED


def load_panel(path: Path | str = PANEL_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def update_manifest(milestone: str, artifacts: dict[str, str]) -> None:
    manifest: dict = {}
    if RUN_MANIFEST.exists():
        manifest = json.loads(RUN_MANIFEST.read_text())
    manifest[milestone] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "seed": SEED,
        "panel_sha256": _sha256(PANEL_PATH),
        "artifacts": artifacts,
    }
    RUN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RUN_MANIFEST.write_text(json.dumps(manifest, indent=2))
