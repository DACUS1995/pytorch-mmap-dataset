from logging import root
from time import time
import os

from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

from pytorch_mmap_dataset import MMAPDataset


DATASET_ROOT_PATH = "./local_test_dir/testSample"

class DiskReadDataset(Dataset):
    def __init__(self, root_path: str) -> None:
        super().__init__()
        self.root_path = root_path
        self.images = os.listdir(root_path) * 10

    def __getitem__(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.root_path, image_name)).convert("RGB")
        image = np.array(image).flatten()
        return image, np.zeros(10)

    def __len__(self):
        return len(self.images)


def benchmark_disk_read(root_dir: str = DATASET_ROOT_PATH):
    start_time = time()
    dataset = DiskReadDataset("./local_test_dir/testSample")

    for idx, (image, label) in enumerate(dataset):
        pass

    duration = time() - start_time
    print(f"Disk read benchmark: {duration} seconds")


def benchmark_in_memory(root_dir: str = DATASET_ROOT_PATH):
    images = []
    labels = []
    dataset = DiskReadDataset("./local_test_dir/testSample")

    for idx, (image, label) in enumerate(dataset):
        images.append(image)
        labels.append(label)
    
    start_time = time()

    for idx, (image, label) in enumerate(zip(images, labels)):
        pass

    duration = time() - start_time
    print(f"In memory benchmark: {duration} seconds")


def benchmark_mmap(root_dir: str = DATASET_ROOT_PATH):
    images = []
    labels = []
    dataset = DiskReadDataset("./local_test_dir/testSample")

    for idx, (image, label) in enumerate(dataset):
        images.append(image)
        labels.append(label)

    dataset = MMAPDataset(images, labels, size=len(dataset))

    start_time = time()

    for idx, (image, label) in enumerate(dataset):
        pass

    duration = time() - start_time
    print(f"MMAP benchmark: {duration} seconds")


def main():
    benchmark_disk_read()
    benchmark_in_memory()
    benchmark_mmap()


if __name__ == "__main__":
    main()