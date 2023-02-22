import os
from typing import List, Tuple

import numpy as np
import torch

from . import base


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    raw = np.load(base.DATA_PATH / "test_data.npz")
    return raw["images"]


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x_min = np.percentile(x, 5.0)
    x_max = np.percentile(x, 99.5)
    return (x - x_min) / (x_max - x_min)


class CellDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath: os.PathLike = base.DATA_PATH / "test_data.npz",
        transform=None,
        target_transform=None,
    ) -> None:
        self.filepath = filepath
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(filepath)
        self.images = data.get("images", None)
        self.labels = data.get("labels", None)

        if self.labels is not None:
            assert self.images.shape[0] == self.labels.shape[0]

    def image_stats(self) -> Tuple[float, float]:
        return np.mean(self.images), np.std(self.images)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image = self.images[idx, ...].astype(np.float32)
        image = torch.Tensor(image[np.newaxis, ...])

        if self.labels is not None:
            label = torch.as_tensor(self.labels[idx])
        else:
            label = None

        if self.transform:
            image = self.transform(image)

        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

    @property
    def shape(self) -> Tuple[int]:
        return self.images.shape[1:]

    def random_batch(self, size: int = 5) -> List[Tuple[int, np.ndarray]]:
        idx = base.RNG.choice(np.arange(len(self)), size=size)

        return [
            base.ImageWithAnnotation(
                data=self.images[i],
                identifier=i,
                label=base.ImageLabel.NOLABEL,
            )
            for i in idx
        ]
