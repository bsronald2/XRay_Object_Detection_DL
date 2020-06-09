import logging
import tensorflow as tf
from pathlib import Path
import os
from src.data import data_utils
from src.data.DataGenerator import DataGenerator
# Helper libraries
import numpy as np


# To Create Synthetic dataset
import imageio
import imgaug as ia


class GDXray:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    DIR = Path('data/raw')
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 64
    tf.random.set_seed(1)
    np.random.seed(1)

    def __init__(self):
        logger = logging.getLogger(__name__)
        logger.info('Init GDXray')
        logger.info(f'Tensor version {tf.__version__}')
        self.classes_name = [i.name for i in GDXray.DIR.glob("*") if i.name != '.gitkeep']
        # Directories
        self.processed_dir = Path('data/processed/')
        self.ptest_dir = self.processed_dir / 'test'
        self.ptrain_dir = self.processed_dir / 'train'
        self.raw_dir = Path(GDXray.DIR)
        # Logs
        logger.info(f'# test images:{len(list(self.ptest_dir.glob("*/*.png")))}')
        logger.info(f'# train images:{len(list(self.ptrain_dir.glob("*/*.png")))}')

    def load_dataset(self):
        """Return:
         Iterators of train and test data-sets.
        """
        train_ds = self.__load_batch(str(self.ptest_dir.absolute() / '*/*.png'))
        test_ds = self.__load_batch(str(self.ptrain_dir.absolute() / '*/*.png'))

        return train_ds, test_ds

    def __load_batch(self, dir, shuffle_buffer_size=2000):
        # Get datasets of all files matching one or more glob patterns.
        list_ds = tf.data.Dataset.list_files(dir)
        # This transformation applies `map_func` to each element of this dataset, and
        # returns a new dataset containing the transformed elements, in the same
        # order as they appeared in the input.
        labeled_ds = list_ds.map(self.__process_path, num_parallel_calls=GDXray.AUTOTUNE)
        prepared_ds = data_utils.prepare_ds(labeled_ds, buffer_size=GDXray.AUTOTUNE,
                                            shuffle_buffer_size=shuffle_buffer_size)

        return prepared_ds

    def pre_process(self, input_dir, output_dir):
        """ Make..."""
        p_raw_ds = self.__load_batch(str(input_dir / '*.png'), shuffle_buffer_size=3000)
        data_generator = DataGenerator()
        data_generator.generate_data_K(p_raw_ds, save_to_dir=output_dir)

    def __get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory

        return parts[-2] == self.classes_name

    def __decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)

        # resize the image to the desired size.
        return tf.image.resize(img, [GDXray.IMG_HEIGHT, GDXray.IMG_WIDTH])

    def __process_path(self, file_path):
        """
        map_func that returns a new transformed elements of the Data-set.
        """
        label = self.__get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.__decode_img(img)

        return img, label
