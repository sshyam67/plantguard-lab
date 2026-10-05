"""Validate PlantVillage and build deterministic group-aware splits."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_leaf_map(repo: Path) -> dict[str, str]:
    path = repo / "leaf_grouping" / "leaf-map.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k).replace("\\", "/"): str(v) for k, v in data.items()}


def choose_splits(groups: dict[str, list[Path]], seed: int) -> dict[Path, str]:
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    result: dict[Path, str] = {}
    total = sum(len(groups[k]) for k in keys)
    targets = {"train": total * 0.70, "val": total * 0.15}
    counts = Counter()
    for key in keys:
        if counts["train"] < targets["train"]:
            split = "train"
        elif counts["val"] < targets["val"]:
            split = "val"
        else:
            split = "test"
        for path in groups[key]:
            result[path] = split
            counts[split] += 1
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("data/raw/PlantVillage-Dataset"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy instead of hard-linking images")
    args = p.parse_args()
    color = args.repo / "raw" / "color"
    if not color.exists():
        raise SystemExit("Dataset missing. Run scripts/download_dataset.py first.")
    class_dirs = sorted(d for d in color.iterdir() if d.is_dir())
    if len(class_dirs) != 38:
        raise SystemExit(f"Expected 38 classes, found {len(class_dirs)}")
    leaf_map = load_leaf_map(args.repo)
    rows: list[dict[str, str]] = []
    summary = Counter()
    for class_dir in class_dirs:
        groups: dict[str, list[Path]] = defaultdict(list)
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel = image.relative_to(args.repo).as_posix()
            group = leaf_map.get(rel, leaf_map.get(image.name, image.stem))
            groups[group].append(image)
        assignments = choose_splits(groups, args.seed)
        crop, disease = class_dir.name.split("___", 1)
        for image, split in assignments.items():
            destination = args.output / split / class_dir.name / image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(image, destination) if args.copy else destination.hardlink_to(image)
            rows.append({
                "path": destination.as_posix(), "class_name": class_dir.name,
                "crop": crop, "disease": disease, "leaf_group": next(
                    k for k, values in groups.items() if image in values
                ), "split": split,
            })
            summary[(class_dir.name, split)] += 1
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "split_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "split", "count"])
        for (label, split), count in sorted(summary.items()):
            writer.writerow([label, split, count])
    print(f"Prepared {len(rows):,} images in {args.output}")


if __name__ == "__main__":
    main()

