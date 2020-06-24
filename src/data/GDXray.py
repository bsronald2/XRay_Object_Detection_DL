import logging
import tensorflow as tf
from pathlib import Path, PurePath
import os
from src.data import data_utils
from src.config import dim
from src.data.DataGenerator import DataGenerator


class GDXray:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    DIR = Path('data/raw')
    IMG_HEIGHT = dim[0]
    IMG_WIDTH = dim[1]
    BATCH_SIZE = 16
    BUFFER_SIZE = 3000

    def __init__(self, input_path, train_val_ds=False):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Init GDXray')
        self.train_val_ds = train_val_ds
        self.input_path = Path(input_path)
        # TODO need to retrieve polygons. The next line is useful for retrieve labels form dir names.
        self.classes_name = [i.name for i in GDXray.DIR.glob("*") if i.name != '.gitkeep']
        # Directories
        self.processed_dir = Path('data/processed/')
        self.raw_dir = Path(GDXray.DIR)

        # Data Augmentation arguments
        self.data_gen_args = dict(rotation_range=45,
                                  width_shift_range=0.33,
                                  height_shift_range=0.33,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True
                                  )

    def load_dataset(self):
        """
        Parameters:
            imgs_dir: a str or Path object type
            train: if param is true load images & label otherwise just images.
        Return:
         A data set Iterator (img, label) or Iterator just imgs
        """
        self.logger.info(f'images size:{len(list(self.input_path.glob("*.png")))}')
        data_set = self.load_batch(str(self.input_path.absolute() / '*.png'))

        return data_set

    def load_img(self):
        return self.process_path(str(self.input_path))

    def load_batch(self, dir, shuffle_buffer_size=2000):
        # Get datasets of all files matching one or more glob patterns.
        list_ds = tf.data.Dataset.list_files(dir)
        # This transformation applies `map_func` to each element of this dataset, and
        # returns a new dataset containing the transformed elements, in the same
        # order as they appeared in the input.
        labeled_ds = list_ds.map(self.process_path, num_parallel_calls=GDXray.AUTOTUNE)
        prepared_ds = GDXray.prepare_ds(labeled_ds, buffer_size=GDXray.AUTOTUNE,
                                            shuffle_buffer_size=shuffle_buffer_size)

        return prepared_ds

    @staticmethod
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

    def pre_process(self, output_dir):
        """ Make..."""
        p_raw_ds = self.load_batch(str(self.input_path / '*.png'), shuffle_buffer_size=GDXray.BUFFER_SIZE)
        final_dir = PurePath(str(self.input_path)).name
        data_generator = DataGenerator()
        # data_generator.generate_data_K(p_raw_ds, save_to_dir=output_dir, prefix=final_dir, data_gen=self.data_gen_args)
        data_generator.generate_data_I(ds=p_raw_ds, save_to_dir=output_dir)

    def get_label(self, file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)

        # The second to last is the class-directory
        return parts[-2] == self.classes_name

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_png(img, channels=0)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        # img = tf.image.convert_image_dtype(img, tf.float32) # model input will transform data.

        # resize the images to the desired size.
        return tf.image.resize(img, [GDXray.IMG_HEIGHT, GDXray.IMG_WIDTH])

    def process_path(self, file_path):
        """
        map_func that returns a new transformed elements of the Data-set.
        """
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        if self.train_val_ds:
            label = self.get_label(file_path)
            return img, label

        return img

test_imgs = '/home/ronald/PycharmProjects/x-ray-deep-learning/X-ray_Object_Detection/data/raw/multi/images/test'
ds = GDXray(test_imgs, train_val_ds=False).load_dataset()
for imgs_batch in ds.as_numpy_iterator():
    print(imgs_batch.shape)