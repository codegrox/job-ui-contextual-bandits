from __future__ import annotations

import argparse
from pathlib import Path

from config import get_config
from src.utils_io import save_json, ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="debug")
    parser.add_argument("--seed", type=int, default=4014)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config(args.preset, args.root)
    paths = cfg["paths"]
    ensure_dirs(paths["raw"])
    save_json(cfg["arm_profiles"], paths["raw"] / "arm_profiles.json")
    print(f"Saved arm profiles to {paths['raw'] / 'arm_profiles.json'}")


if __name__ == "__main__":
    main()
