import numpy as np
import tensorflow as tf
import json
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from tensorflow.keras.utils import Sequence
from imgaug.augmentables import Keypoint, KeypointsOnImage

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    iaa.GaussianBlur(sigma=(0, 0.5)),
    # Crop image with random from 0 to 10%
    # But we only crop about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.Crop(percent=(0, 0.1), keep_size=True)),
    # Strengthen or weaken the contrast in each images.
    iaa.LinearContrast((0.75, 1)),

    # Add gaussian noise.
    # For 30% of all images, we sample the noise once per pixel.
    # For the other 30% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.3),

    # Apply affine transformations to each images.
    # Scale/zoom them.
    iaa.Affine(
        scale={"x": (1.0, 1.1), "y": (1.0, 1.1)})
], random_order=True)  # apply augmenters in random order


class GDXrayDataGenerator(Sequence):
    # Generates data for Keras
    def __init__(self, image_paths, ann_path, labels, n_classes, batch_size=32, dim=(256, 256, 3),
                 shuffle=True, augment=False): # add synthetic param

        # Annotation image info
        self.ann_path = ann_path
        self.labels = labels
        self.dim = dim
        self.n_classes = n_classes
        self.image_paths = image_paths
        self.indexes = np.arange(len(self.image_paths)) # Do nothing for shuffle
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.image_paths) / self.batch_size)) # Multiply by the size batch or remove batch

    def __getitem__(self, index):
        # Generate indexes of the batch
        # if index 0 and batch 4 in range(0, 17) retrieve values [0 1 2 3]
        # if index 1 and batch 4 in range(0, 17) retrieve values [4 5 6 7]
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # image_paths = [self.image_paths[k] for k in indexes]
        # # annot_paths = [self.annot_paths[k] for k in indexes]
        # # TODO just send an image to create multiple images on the fly using imgaug lib until the size of the batch
        # # self.image_paths[index]
        # X, y = self.__data_generation(image_paths)

        index_aux = self.indexes[index]
        img_path = self.image_paths[index_aux]
        X, y = self.__data_generation(img_path)

        return X, y

    def set_ann_data(self):
        with open(self.ann_path, 'r') as read_it:
            ann_data = json.load(read_it)
        self.dict_imgs = ann_data.get('images')
        self.dict_ann = ann_data.get('annotations')
        self.dict_cat = ann_data.get('categories')

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __search_array(self, array, key, value):
        return next((obj for obj in array if obj[key] == value), None)  # return object

    def get_img_info(self, img_name):
        """
        return img_label and segmentation points of the image as tuple
        """
        img_obj = self.__search_array(self.dict_imgs, 'file_name', img_name)
        if img_obj is not None:
            ann_obj = self.__search_array(self.dict_ann, 'image_id', str(img_obj['id']))
            if ann_obj is not None:
                kps = self.__get_img_seg_kps(ann_obj['segmentation'])
                label = self.__search_array(self.dict_cat, 'id', ann_obj['category_id'])
                return label['name'], kps

        return None

    def __get_img_seg_kps(self, img_seg):
        points = list()
        # iterate every two steps due to json array with segmented
        # points are in the following way: [x1,y1,x2,y2,..,..,xn,yn]
        for i in range(0, len(img_seg), 2):  # iterate every two steps
            chunk = img_seg[i:i + 2]
            points.append(Keypoint(x=chunk[0], y=chunk[1]))

        return points

    def get_augimg(self, img, img_info):
        label, points = img_info
        kps = KeypointsOnImage(points, shape=img.shape)
        if img.shape != self.dim:
            img = ia.imresize_single_image(img, self.dim[0:2])
            kps = kps.on(img)
        # Augment keypoints and images.
        img_aug, kps_aug = seq(image=img, keypoints=kps)
        aug_points = [[kp.x, kp.y] for kp in kps_aug.keypoints]
        aug_points_dic = {'label': label, 'points': aug_points}

        return img_aug, aug_points_dic

    def get_mask(self, img, imgaug_shape):
        blank = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.float32)
        points = np.array(imgaug_shape['points'], dtype=np.int32)
        cv2.fillPoly(blank, [points], 255)
        blank = blank / 255.0

        return np.expand_dims(blank, axis=2)

    def get_poly(self, annot_path):
        # reads in shape_dicts
        with open(annot_path) as handle:
            data = json.load(handle)
        shape_dicts = data['shapes']

        return shape_dicts

    def __data_generation(self, img_path):
        X = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_classes), dtype=np.float32)

        # retrieve img as numpy
        img = cv2.imread(str(img_path))  # our images are gray_scale
        img = (img / 255.0).astype(np.float32)
        images = [np.copy(img) for _ in range(self.batch_size)]
        img_info = self.get_img_info(img_path.name)
        for i, image in enumerate(images):
            imgaug, imgaug_shape = self.get_augimg(img, img_info)
            imgaug_mask = self.get_mask(imgaug, imgaug_shape)
            X[i,] = imgaug
            y[i,] = imgaug_mask

        return X, y
