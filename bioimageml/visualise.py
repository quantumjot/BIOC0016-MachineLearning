from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from skimage.util import montage
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from .base import ImageLabel, ImagePrediction, ImageWithAnnotation
from .dataset import CellDataset

LABELS = [
    label.name.capitalize()
    for label in ImageLabel
    if (label.value >= 0 and label.value <= 5)
]


def visualize_predictions(
    images: List[ImageWithAnnotation], predictions: List[ImagePrediction]
):

    num_images = min(len(images), 5)

    for img, pred in zip(images[:num_images], predictions[:num_images]):
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        fig.tight_layout(pad=5.0)
        ax[0].imshow(img.data)
        ax[0].set_axis_off()
        ax[0].set_title(f"Image: {img.identifier}, label: {img.label.name}")

        p_idx = np.argsort(pred.logits)

        ax[1].barh(
            np.arange(pred.logits.shape[0]),
            softmax(pred.logits[p_idx]),
            align="center",
            height=0.75,
            tick_label=[LABELS[i] for i in p_idx],
            edgecolor="k",
            linewidth=2,
            color=["w", "w", "w", "w", "b"],
            log=True,
        )
        ax[1].set_title(f"Prediction: {pred.label.name}")
        ax[1].set_xlabel(r"$ \log_{10} P(label~\vert~data)$")


def visualize_confusion_matrix(
    images: List[ImageWithAnnotation], predictions: List[ImagePrediction]
):

    full_labels = [label.name for label in ImageLabel]

    cm = confusion_matrix(
        [img.label.name for img in images],
        [pred.label.name for pred in predictions],
        labels=full_labels,
    )

    confmat = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=full_labels,
    )
    confmat.plot(
        xticks_rotation="vertical",
    )
    plt.show()


def visualize_report(
    images: List[ImageWithAnnotation], predictions: List[ImagePrediction]
):
    full_labels = [label.name for label in ImageLabel]

    report = classification_report(
        [img.label.name for img in images],
        [pred.label.name for pred in predictions],
        labels=full_labels,
        zero_division=0,
    )

    print(report)


def visualize_random_batch(dataset: CellDataset, *, batch_size: int = 10):
    """Plot a montage of a random batch."""
    if batch_size < 1:
        raise ValueError("`batch_size` should be greater than one.")

    batch = dataset.random_batch(batch_size)
    cols = 8
    rows = max(1, np.ceil(batch_size / cols))
    m = montage(
        np.stack([b.data for b in batch], axis=0),
        rescale_intensity=True,
        grid_shape=(rows, cols),
        padding_width=3,
        fill=255,
    )
    fig, ax = plt.subplots(figsize=(16, 4 * rows))
    ax.imshow(m, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()


def visualise_outputs(outputs: np.ndarray) -> None:
    """Visualise the outputs of the model."""

    cols = 8
    rows = max(1, np.ceil(outputs.shape[1] / cols))

    m = montage(
        outputs[0, ...],
        rescale_intensity=True,
        grid_shape=(rows, cols),
        padding_width=3,
        fill=1.0,
    )
    fig, ax = plt.subplots(figsize=(16, 4 * rows))
    ax.imshow(m, cmap=plt.cm.gray)
    ax.set_axis_off()
    plt.show()
