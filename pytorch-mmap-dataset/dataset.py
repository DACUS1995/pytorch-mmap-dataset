import gc
import os
from typing import Iterable, Tuple

import numpy as np
from torch.utils.data.dataset import Dataset

DEFAULT_INPUT_FILE_NAME = "input.data"
DEFAULT_LABELS_FILE_NAME = "labels.data"


class MMAPDataset(Dataset):
    def __init__(
        self,
        input_iter: Iterable[np.ndarray],
        labels_iter: Iterable[np.ndarray],
        mmap_path: str = None,
        size: int = None,
    ) -> None:
        super().__init__()

        self.mmap_inputs: np.ndarray = None
        self.mmap_labels: np.ndarray = None

        if mmap_path is None:
            mmap_path = os.path.abspath(os.getcwd())
        self._mkdir(mmap_path)

        self.mmap_input_path = os.path.join(mmap_path, DEFAULT_INPUT_FILE_NAME)
        self.mmap_labels_path = os.path.join(mmap_path, DEFAULT_LABELS_FILE_NAME)

        # If the total size is not known we load the dataset in memory first
        if size is None:
            inputs = []
            labels = []

            for idx, (input, label) in enumerate(zip(input_iter, labels_iter)):
                inputs.append(input)
                labels.append(label)

            if len(inputs) != len(labels):
                raise Exception(
                    f"Inputs samples count {len(inputs)} is different than the labels count {len(labels)}"
                )

            if not isinstance(inputs[0], np.ndarray):
                raise TypeError("Inputs and labels must be of type np.ndarray")

            input_iter = inputs
            labels_iter = labels
            size = len(inputs)

        self.length = size

        for idx, (input, label) in enumerate(zip(input_iter, labels_iter)):
            if self.mmap_inputs is None:
                self.mmap_inputs = self._init_mmap(
                    self.mmap_input_path, input.dtype, (self.length, *input.shape)
                )
                self.mmap_labels = self._init_mmap(
                    self.mmap_labels_path, label.dtype, (self.length, *label.shape)
                )

            self.mmap_inputs[idx][:] = input[:]
            self.mmap_labels[idx][:] = label[:]

        del inputs
        del labels
        gc.collect()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray]:
        return self.mmap_inputs[idx], self.mmap_labels[idx]

    def __len__(self) -> int:
        return self.length

    def _mkdir(self, path: str) -> None:
        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return
        except:
            raise ValueError(
                "Failed to create the path (check the user write permissions)."
            )

    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int]) -> np.ndarray:
        return np.memmap(
            path,
            dtype=dtype,
            mode="w+",
            shape=shape,
        )
