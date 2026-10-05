"""Train a reproducible MobileNetV3-Small baseline."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("artifacts"))
    args = p.parse_args()
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from src.model import build_model

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15, 0.15, 0.10, 0.05),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    sets = {name: datasets.ImageFolder(args.data_dir / name, transform=train_tf if name == "train" else eval_tf)
            for name in ("train", "val", "test")}
    if sets["train"].classes != sets["val"].classes or sets["train"].classes != sets["test"].classes:
        raise SystemExit("Class ordering differs across splits.")
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=name == "train",
                                num_workers=args.num_workers, pin_memory=device == "cuda",
                                persistent_workers=args.num_workers > 0)
               for name, ds in sets.items()}
    model = build_model(len(sets["train"].classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best = [], -1.0
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        record = {"epoch": epoch}
        for phase in ("train", "val"):
            model.train(phase == "train")
            total_loss = correct = count = 0
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == "train"):
                    logits = model(inputs)
                    loss = loss_fn(logits, labels)
                    if phase == "train":
                        loss.backward(); optimizer.step()
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                count += labels.size(0)
            record[f"{phase}_loss"] = total_loss / count
            record[f"{phase}_accuracy"] = correct / count
        history.append(record)
        print(json.dumps(record))
        if record["val_accuracy"] > best:
            best = record["val_accuracy"]
            torch.save({"model_state": model.state_dict(), "classes": sets["train"].classes,
                        "seed": args.seed, "epoch": epoch, "val_accuracy": best},
                       args.output / "best_model.pt")
    with (args.output / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    print(f"Best validation accuracy: {best:.4f}; checkpoint: {args.output / 'best_model.pt'}")


if __name__ == "__main__":
    main()
