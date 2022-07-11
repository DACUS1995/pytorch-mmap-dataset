import numpy as np
from dataset import MMAPDataset


def main():
    inputs = [np.ones(10) for _ in range(10)]
    labels = [np.ones(10) for _ in range(10)]

    dataset = MMAPDataset(inputs, labels)

    print(dataset.length)
    print(dataset[0][0].dtype)
    print(dataset[0][0].shape)


if __name__ == "__main__":
    main()
