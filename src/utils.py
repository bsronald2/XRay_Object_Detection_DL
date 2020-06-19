import os
from pathlib import Path


def create_dir(dir):
    if Path.exists(dir):
        for item_path in Path.iterdir(dir):
            Path.unlink(item_path)
    else:
        os.makedirs(dir)


def delete_file(file_path):
    if type(file_path) is str:
        file_path = Path(file_path)
    if file_path.exists():
        Path.unlink(file_path)
