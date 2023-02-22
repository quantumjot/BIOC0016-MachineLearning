import dataclasses
import enum
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

RNG = np.random.default_rng()
DATA_PATH = Path(__file__).parent / "data"
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_MEAN_STD = (45.1938853515625, 39.78675684923092)
CSS_PATH = Path(__file__).parent.parent / "files" / "style.css"


class ImageLabel(enum.Enum):
    NOLABEL = -1
    INTERPHASE = 0
    PROMETAPHASE = 1
    METAPHASE = 2
    ANAPHASE = 3
    APOPTOSIS = 4
    UNKNOWN = 99


@dataclasses.dataclass
class ImagePrediction:
    label: ImageLabel
    logits: np.ndarray


@dataclasses.dataclass
class ImageWithAnnotation:
    data: np.ndarray
    identifier: int
    label: ImageLabel

    @property
    def hash(self) -> str:
        m = hashlib.md5(self.data.tostring())
        return m.hexdigest()


def _set_css_style(css_file_path: os.PathLike = CSS_PATH):
    """Read the custom CSS file and load it into Jupyter."""

    from IPython.core.display import HTML

    with open(css_file_path, "r") as css_file:
        styles = css_file.read()
    return HTML(f"<style>{styles}</style>")
