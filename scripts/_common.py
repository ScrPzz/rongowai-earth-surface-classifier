"""Small helpers shared by the numbered scripts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rongowai_ddm.config import Config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "default.yaml", help="YAML configuration"
    )
    return p


def load_config(args: argparse.Namespace) -> Config:
    return Config.load(args.config)


class Encoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, cls=Encoder)
    print(f"wrote {path}")


def load_json(path: str | Path):
    with open(path) as fh:
        return json.load(fh)


class Timer:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        print(f"[{self.label}] {self.seconds / 60:.1f} min")
