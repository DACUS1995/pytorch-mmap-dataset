import gc
import os
from typing import Any, Callable, Iterable, List, Tuple, Union

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch


DEFAULT_INPUT_FILE_NAME = "input.data"
DEFAULT_LABELS_FILE_NAME = "labels.data"


class MMAPDataset(Dataset):
    def __init__(
        self,
        input_iter: Iterable[np.ndarray],
        labels_iter: Iterable[np.ndarray],
        mmap_path: str = None,
        size: int = None,
        transform_fn: Callable[..., Any] = None,
        
    ) -> None:
        super().__init__()

        self.mmap_inputs: np.ndarray = None
        self.mmap_labels: np.ndarray = None
        self.transform_fn = transform_fn

        if mmap_path is None:
            mmap_path = os.path.abspath(os.getcwd())
        self._mkdir(mmap_path)

        self.mmap_input_path = os.path.join(mmap_path, DEFAULT_INPUT_FILE_NAME)
        self.mmap_labels_path = os.path.join(mmap_path, DEFAULT_LABELS_FILE_NAME)

        # If the total size is not known we load the dataset in memory first
        if size is None:
            input_iter, labels_iter = self._consume_iterable(input_iter, labels_iter)
            size = len(input_iter)

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

        del input_iter
        del labels_iter
        gc.collect()


    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        if self.transform_fn:
            return self.transform_fn(self.mmap_inputs[idx]), torch.tensor(self.mmap_labels[idx]) 
        return self.mmap_inputs[idx], self.mmap_labels[idx]


    def __len__(self) -> int:
        return self.length


    def _consume_iterable(self, input_iter: Iterable[np.ndarray], labels_iter: Iterable[np.ndarray]) -> Tuple[List[np.ndarray]]:
        inputs = []
        labels = []

        for idx, (input, label) in enumerate(zip(input_iter, labels_iter)):
            inputs.append(input)
            labels.append(label)

        if len(inputs) != len(labels):
            raise Exception(
                f"Input samples count {len(inputs)} is different than the labels count {len(labels)}"
            )

        if not isinstance(inputs[0], np.ndarray):
            raise TypeError("Inputs and labels must be of type np.ndarray")

        return inputs, labels


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


    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int], remove_existing: bool = False) -> np.ndarray:
        open_mode = "r+"

        if remove_existing:
            open_mode = "w+"
        
        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )
