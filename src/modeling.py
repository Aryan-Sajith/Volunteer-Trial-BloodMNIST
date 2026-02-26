from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {name}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
