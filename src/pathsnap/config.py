"""Load configs/config.yaml with paths resolved against the repo root."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {k: (ROOT / v) for k, v in cfg.get("paths", {}).items()}
    cfg["root"] = ROOT
    return cfg
