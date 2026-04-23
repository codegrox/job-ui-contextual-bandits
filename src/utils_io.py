from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dirs(path.parent)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_pickle(path)


def load_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def save_json(obj: Any, path: Path) -> None:
    ensure_dirs(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_log(path: Path, message: str) -> None:
    ensure_dirs(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")
