import os
from pathlib import Path


def create_dir(dir):
    if Path.exists(dir):
        for item_path in Path.iterdir(dir):
            Path.unlink(item_path)
    else:
        os.makedirs(dir)
