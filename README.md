# pytorch-mmap-dataset

A custom pytorch Dataset extension that provides a faster iteration and better main memory usage when going over a dataset by using a memory mapped file to store amy potential big files that would normally be read on demand.

---

By using a generic iterable the dataset can support any type of file reading operations when creating the memory mapped file. 

For the best memory wise performance the usage of a generator is recommended, an example for images is provided below:

```python
size_dataset = None
def image_iter(root_path: str = DATASET_ROOT_PATH):
    images = os.listdir(root_path)
    global size_dataset
    size_dataset = len(images)
    for image_name in images:
        image = Image.open(os.path.join(root_path, image_name)).convert("RGB")
        image = np.array(image).flatten()
        yield image
```

By providing the size argument the files will be read one by one thus avoiding unnecesary memory usage when copying the data in the mmap file.
```python
dataset = MMAPDataset(image_iter(), image_iter(), size=size_dataset)

for idx, (input, label) in enumerate(dataset):
    # The model goes brrr
```

---

For a benchmark comparing a normally reading dataset, an in-memory dataset and mmap dataset you can check/run the `benchmark.py` file.