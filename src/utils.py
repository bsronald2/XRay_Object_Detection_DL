import os
import random
import os
from pathlib import Path
from matplotlib import pyplot as plt
from src.config import reports_path
from datetime import datetime

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


def create_random_list_of_size(files, size):
    files_len = len(files)
    random_list = [files[random.randrange(files_len)] for _ in range(0, size)]

    return random_list


def save_model_history(data):
    # Plot image
    plt.plot(data['y'])
    plt.plot(data['X'])
    plt.title(data['title'])
    plt.ylabel(data['ylabel'])
    plt.xlabel(data['xlabel'])
    plt.legend(data['legend'], loc='upper left')

    # Save Image
    time_stamp = datetime.timestamp(datetime.now())
    plt.savefig(f"{reports_path}/{data['title']}_{time_stamp}.png")
    plt.clf()
    plt.close()


def save_iou_th(thresholds, ious, threshold_best, iou_best):
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()

    # Save Image
    time_stamp = datetime.timestamp(datetime.now())
    plt.savefig(f"{reports_path}/iou_vs_thr_{time_stamp}.png")
    plt.clf()
    plt.close()


def find_first(array, key, value):
    return next((obj for obj in array if obj[key] == value), None)  # return object


def find_all(array, key, value):
    return [obj for obj in array if obj[key] == value]
