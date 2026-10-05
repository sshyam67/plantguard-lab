"""Evaluate clean and degraded test images and save tidy metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from src.degradations import degrade


class AllConditionsDataset:
    def __init__(self, samples, transform, conditions):
        self.samples = samples
        self.transform = transform
        self.conditions = conditions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        variants = []
        for kind, severity in self.conditions:
            variant = image if kind == "clean" else degrade(image, kind, severity)
            variants.append(self.transform(variant))
        return torch.stack(variants), label


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--model", type=Path, default=Path("artifacts/best_model.pt"))
    p.add_argument("--output", type=Path, default=Path("artifacts/robustness_results.csv"))
    args = p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from src.model import load_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_checkpoint(args.model, device)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_dir / "test")
    if dataset.classes != checkpoint["classes"]:
        raise SystemExit("Checkpoint and dataset class order do not match.")
    conditions = [("clean", 0)] + [(kind, severity)
        for kind in ("blur", "dark", "bright", "noise", "resolution", "jpeg")
        for severity in (1, 2, 3)]
    correct = [0 for _ in conditions]
    confidence = [0.0 for _ in conditions]
    count = 0
    all_conditions = AllConditionsDataset(dataset.samples, transform, conditions)
    loader = DataLoader(all_conditions, batch_size=16, num_workers=4,
                        pin_memory=device == "cuda", persistent_workers=True)
    with torch.no_grad():
        for variants, labels in loader:
            labels = labels.to(device)
            for index in range(len(conditions)):
                probabilities = model(variants[:, index].to(device)).softmax(1)
                correct[index] += (probabilities.argmax(1) == labels).sum().item()
                confidence[index] += probabilities.max(1).values.sum().item()
            count += labels.size(0)
    rows = []
    for index, (kind, severity) in enumerate(conditions):
        rows.append({"condition": kind, "severity": severity, "n": count,
                     "accuracy": correct[index] / count,
                     "mean_confidence": confidence[index] / count})
        print(rows[-1])
    clean_accuracy = rows[0]["accuracy"]
    for row in rows:
        row["accuracy_drop"] = clean_accuracy - row["accuracy"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
