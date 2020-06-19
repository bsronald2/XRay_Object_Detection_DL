####################################
# GDXray.load_dataset
####################################
# for images, label in test_labeled_ds.take(1):
#     print("Image shape: ", images.numpy().shape)
#     print("Label: ", label.numpy())
#     index = next((i for i, j in enumerate(label) if j), None)
#     data_utils.show(images, self.classes_name[index])

# image_ds = tfds.as_numpy(test_ds)
# for i, l in image_ds:
#     print(i.shape)
#     print(l.shape)
#     break
# image_batch, label_batch = next(iter(train_ds))
# print(image_batch.shape)
# print(label_batch)

#list(dataset.as_numpy_iterator())
# p_raw_ds.element_spec
########################################
# Split training,test, validation imgs
########################################
 #split_folders.ratio(raw_imagepath, output=output_filepath, seed=1337, ratio=(.8, .1, .1))  # default values

##############################################33
# ia.imshow(
#     ia.draw_grid(
#         imgs[0] +
#         imgs[1] +
#         imgs[2] +
#         imgs[3],
#         # images_deterministic[0] + [whitespace] + images_deterministic[1],  # second row
#         rows=5,
#         cols=10,
#     )
# )
######################################################

import numpy as np
import os
from pathlib import Path
from src.data.GDXrayDataGenerator import GDXrayDataGenerator
# ROOT_DIR = os.path.dirname(os.path.abspath('src'))
# print(ROOT_DIR)
ROOT = Path('../../data/raw')
images_path = ROOT / 'images'
images_path = images_path.resolve()
print(images_path)
imgs_paths = sorted([i.absolute() for i in images_path.glob("*.png") if i.is_file()])
ann_path = str(ROOT/ 'annotation')
print(imgs_paths)
labels = ['gun', 'knife', 'shuriken', 'razor_blade']
dim = (256, 256, 3)
# a = GDXrayDataGenerator(imgs_paths, ann_path, labels, len(labels), dim=dim, augment=True)
indexes = np.arange(len(imgs_paths))

batch_size = 4
index = 1
# if index 0 and batch 4 in range(0, 17) retrieve values [0 1 2 3]
# if index 1 and batch 4 in range(0, 17) retrieve values [4 5 6 7]
indexes = indexes[index * batch_size:(index + 1) * batch_size]
n_classes = len(labels)
X = np.empty((batch_size, *dim), dtype=np.float32)
y = np.empty((batch_size, dim[0], dim[1], n_classes), dtype=np.float32)
# print(indexes)

#
# def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
#     """
#     Creates a list of random minibatches from (X, Y)
#
#     Arguments:
#     X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
#     Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
#     mini_batch_size - size of the mini-batches, integer
#     seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
#
#     Returns:
#     mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
#     """
#
#     m = X.shape[0]  # number of training examples
#     mini_batches = []
#     np.random.seed(seed)
#
#     # Step 1: Shuffle (X, Y)
#     permutation = list(np.random.permutation(m))
#     shuffled_X = X[permutation, :, :, :]
#     shuffled_Y = Y[permutation, :]
#
#     # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
#     num_complete_minibatches = math.floor(
#         m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
#     for k in range(0, num_complete_minibatches):
#         mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
#         mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
#
#     # Handling the end case (last mini-batch < mini_batch_size)
#     if m % mini_batch_size != 0:
#         mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
#         mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
#         mini_batch = (mini_batch_X, mini_batch_Y)
#         mini_batches.append(mini_batch)
#
#     return mini_batches
#
#
# def convert_to_one_hot(Y, C):
#     Y = np.eye(C)[Y.reshape(-1)].T
#     return Y
