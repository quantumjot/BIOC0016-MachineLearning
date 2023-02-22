import os
from typing import List, Optional

import numpy as np
import torch
from torchvision import transforms

from .base import (
    DATA_PATH,
    IMAGE_MEAN_STD,
    TORCH_DEVICE,
    ImageLabel,
    ImagePrediction,
    ImageWithAnnotation,
)


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels: int, output_classes: int) -> None:
        super(SimpleCNN, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.2),
            torch.nn.Linear(128, output_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_layer_activations(
        self, x: torch.Tensor, *, layer: int = 3
    ) -> torch.Tensor:
        """Get the layer activations for part of the model."""
        if layer < 1 or layer > 3:
            raise ValueError("`layer` must be between 1 and 3")
        return self.model[: layer * 3](x)


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    *,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 20,
    filepath: os.PathLike = DATA_PATH / "model.pt",
    learning_rate: float = 0.001,
) -> None:
    """Train the model."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=5e-4),
    )

    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for img, label in train_loader:
            out = model(img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(loader):
        model.eval()

        correct = 0
        for img, label in loader:
            out = model(img)
            pred = out.argmax(
                dim=-1
            )  # Use the class with highest probability.
            correct += int((pred == label).sum())
        return correct / len(loader.dataset)

    for epoch in range(1, epochs):
        train()
        train_acc = test(train_loader)
        if test_loader is not None:
            test_acc = test(test_loader)
        else:
            test_acc = 0.0

        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f},"
            f" Test Acc: {test_acc:.4f}"
        )

    torch.save(model, filepath)


def load_model(
    filepath: os.PathLike = DATA_PATH / "model.pt",
) -> torch.nn.Module:
    """Load a pre-trained model."""
    model = torch.load(filepath, map_location=TORCH_DEVICE)
    model.eval()
    return model


def predict(
    model: torch.nn.Module,
    images: List[ImageWithAnnotation],
) -> List[ImageLabel]:
    """Inference from model."""

    x = torch.from_numpy(
        np.stack([img.data[np.newaxis, ...] for img in images], axis=0).astype(
            np.float32
        ),
    )
    norm = transforms.Normalize(*IMAGE_MEAN_STD)
    x = norm(x)
    x = x.to(TORCH_DEVICE)
    logits = model(x).detach().cpu().numpy()
    labels = np.argmax(logits, axis=1)

    results = [
        ImagePrediction(ImageLabel(lbl), lgt)
        for lbl, lgt in zip(labels, logits)
    ]

    return results


def inspect(
    model: torch.nn.Module,
    images: List[ImageWithAnnotation],
    *,
    layer: int = 1,
    plot: bool = True,
) -> np.ndarray:
    """Inference from model."""

    layer = int(max(min(layer, 2), 1))

    x = torch.from_numpy(
        np.stack([img.data[np.newaxis, ...] for img in images], axis=0).astype(
            np.float32
        ),
    )
    norm = transforms.Normalize(*IMAGE_MEAN_STD)
    x = norm(x)
    x = x.to(TORCH_DEVICE)
    x = model.get_layer_activations(x, layer=layer)
    return x.detach().cpu().numpy()
