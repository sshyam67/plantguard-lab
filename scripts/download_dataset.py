"""Download the maintained PlantVillage source repository."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/PlantVillage-Dataset"))
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        print(f"Dataset already exists: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", str(args.depth), URL, str(args.output)],
        check=True,
    )
    print(f"Downloaded PlantVillage to {args.output}")


if __name__ == "__main__":
    main()

