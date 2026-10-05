"""MobileNetV3 model definition and inference wrapper."""
from __future__ import annotations

import csv
from pathlib import Path


def class_names(path: Path = Path("data/plantvillage_classes.csv")) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [row["class_name"] for row in csv.DictReader(f)]


def build_model(num_classes: int = 38, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_checkpoint(path: Path, device: str = "cpu"):
    import torch
    model = build_model(pretrained=False)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint

