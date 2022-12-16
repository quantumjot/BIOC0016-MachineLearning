import enum
from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "test_data.npz"


class ImageLabel(enum.Enum):
    INTERPHASE = 0
    PROMETAPHASE = 1
    METAPHASE = 2
    ANAPHASE = 3
    APOPTOSIS = 4
    UNKNOWN = 99


def load_data() -> tuple[np.ndarray, np.ndarray]:
    raw = np.load(DATA_PATH)
    return raw["images"]


def normalize(x: np.ndarray) -> np.ndarray:
    x_min = np.percentile(x, 5.0)
    x_max = np.percentile(x, 99.5)
    return (x - x_min) / (x_max - x_min)


class Dataset:
    def __init__(self):
        self.images = load_data()

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.images[idx, ...]

    def __len__(self):
        return self.images.shape[0]

    @property
    def shape(self):
        return self.images.shape[1:]


if __name__ == "__main__":
    d = Dataset()
    print(d.shape)
