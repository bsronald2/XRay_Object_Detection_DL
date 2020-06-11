import matplotlib.pyplot as plt
import os
from pathlib import Path


def prepare_ds(ds, cache=True, shuffle_buffer_size=2000, batch=32, buffer_size=-1):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    # Note: `cache` will produce exactly the same elements during each iteration
    #     through the dataset. If you wish to randomize the iteration order, make sure
    #     to call `shuffle` *after* calling `cache`.
    # Randomly shuffles the elements of this dataset.
    # For perfect shuffling, a buffer size greater than or equal to the
    # full size of the dataset is required.
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    # ds = ds.repeat()

    ds = ds.batch(batch)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=buffer_size)

    return ds


def show(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(label)
    plt.axis('off')
    plt.show()


def create_dir(dir):
    if Path.exists(dir):
        for item_path in Path.iterdir(dir):
            Path.unlink(item_path)
    else:
        os.makedirs(dir)
