from collections import Counter
from typing import Dict

from ipyannotations import images

from ..api import submit_annotation
from ..base import ImageLabel
from ..dataset import CellDataset


class BaseLabeller:
    def __init__(self, dataset: CellDataset) -> None:
        self.dataset = dataset
        self.widget = None
        self.annotations: list = []
        self._current_image = None

    def next(self) -> None:
        self._current_image = self.dataset.random_batch(1)[0]

    def start(self) -> None:
        if self.widget is not None:
            self.next()
            self.widget.on_submit(self.submit)
            self.widget.display(self._current_image.data)
        else:
            raise Exception

    def submit(self, annotation: str) -> None:

        if annotation is None:
            annotation = "UNKNOWN"

        assert self._current_image.label == ImageLabel.NOLABEL
        self._current_image.label = ImageLabel[annotation.upper()]
        self.annotations.append(self._current_image)

        submit_annotation(self._current_image)

        try:
            self.next()
            self.widget.display(self._current_image.data)
        except IndexError:
            pass

    def statistics(self) -> Dict[str, int]:
        stats = Counter([a.label.name for a in self.annotations])
        return dict(stats)


class MitoticInstanceLabeller(BaseLabeller):
    """Label mitotic cells."""

    def __init__(self, *args):
        super().__init__(*args)
        options = [
            label.name.title()
            for label in ImageLabel
            if label != ImageLabel.NOLABEL
        ]
        self.widget = images.ClassLabeller(
            options=options,
            allow_freetext=False,
        )


class MitoticSequenceLabeller:
    """Label mitotic cells."""
