"""Model registry for image classification tasks."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _image_shape(config: dict[str, Any]) -> tuple[int, int, int]:
    data_cfg = config.get('data', {})
    shape = data_cfg.get('image_shape') or config.get('model', {}).get('image_shape')
    if shape is None:
        dataset_name = str(data_cfg.get('dataset_name', 'mnist')).lower()
        return (1, 28, 28) if dataset_name == 'mnist' else (3, 32, 32)
    channels, height, width = [int(value) for value in shape]
    return channels, height, width


class FlattenClassifier(nn.Module):
    """A small MLP baseline for image classification."""

    def __init__(self, image_shape: tuple[int, int, int], num_classes: int, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        channels, height, width = image_shape
        features = channels * height * width
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallConvClassifier(nn.Module):
    """Compact CNN baseline that works for MNIST and CIFAR-10."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        width = int(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class MediumConvClassifier(nn.Module):
    """Deeper CNN baseline for MNIST and CIFAR-10."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        width = int(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class LargeConvClassifier(nn.Module):
    """High-capacity CNN baseline for MNIST and CIFAR-10."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        width = int(hidden_channels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width * 4, width * 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, width * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm buffers for communication analysis."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResidualStageClassifier(nn.Module):
    """ResNet-style classifier sized to mirror the small/medium/large CNN families."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        stage_channels: list[int],
        stage_blocks: list[int],
        head_hidden_dims: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(stage_channels) != len(stage_blocks):
            raise ValueError('stage_channels and stage_blocks must have the same length')
        stem_channels = stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        current_channels = stem_channels
        for stage_index, (out_channels, num_blocks) in enumerate(zip(stage_channels, stage_blocks)):
            stride = 1 if stage_index == 0 else 2
            blocks: list[nn.Module] = [ResidualBlock(current_channels, out_channels, stride=stride)]
            for _ in range(1, int(num_blocks)):
                blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
            stages.append(nn.Sequential(*blocks))
            current_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        head_layers: list[nn.Module] = [nn.Flatten(), nn.Dropout(dropout)]
        head_in = current_channels
        for hidden_dim in head_hidden_dims:
            head_layers.extend([nn.Linear(head_in, int(hidden_dim)), nn.ReLU(), nn.Dropout(dropout)])
            head_in = int(hidden_dim)
        head_layers.append(nn.Linear(head_in, int(num_classes)))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        return self.head(x)


def _build_resnet_variant(name: str, *, in_channels: int, num_classes: int, hidden_channels: int, dropout: float) -> nn.Module:
    width = int(hidden_channels)
    variants = {
        'small_resnet': dict(stage_channels=[width, width * 2], stage_blocks=[1, 1], head_hidden_dims=[]),
        'medium_resnet': dict(stage_channels=[width, width * 2, width * 4], stage_blocks=[1, 1, 1], head_hidden_dims=[width * 2]),
        'large_resnet': dict(stage_channels=[width, width * 2, width * 4, width * 8], stage_blocks=[1, 1, 1, 1], head_hidden_dims=[width * 4, width * 2]),
    }
    spec = variants[name]
    return ResidualStageClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        stage_channels=spec['stage_channels'],
        stage_blocks=spec['stage_blocks'],
        head_hidden_dims=spec['head_hidden_dims'],
        dropout=dropout,
    )


def build_model(config: dict[str, Any]) -> nn.Module:
    """Build a classification model from ``config['model']``."""

    model_cfg = config.get('model', {})
    image_shape = _image_shape(config)
    in_channels = int(image_shape[0])
    num_classes = int(config.get('data', {}).get('num_classes', model_cfg.get('num_classes', 10)))
    name = str(model_cfg.get('name', 'small_cnn')).lower()
    if name in {'small_cnn', 'cnn', 'convnet'}:
        return SmallConvClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=int(model_cfg.get('hidden_channels', 32)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
    if name == 'medium_cnn':
        return MediumConvClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=int(model_cfg.get('hidden_channels', 32)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
    if name == 'large_cnn':
        return LargeConvClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=int(model_cfg.get('hidden_channels', 32)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
    if name in {'small_resnet', 'medium_resnet', 'large_resnet'}:
        return _build_resnet_variant(
            name,
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_channels=int(model_cfg.get('hidden_channels', 32)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
    if name in {'mlp', 'flatten'}:
        return FlattenClassifier(
            image_shape=image_shape,
            num_classes=num_classes,
            hidden_size=int(model_cfg.get('hidden_size', 128)),
            dropout=float(model_cfg.get('dropout', 0.1)),
        )
    raise ValueError(f"Unknown classification model name: {name}")
