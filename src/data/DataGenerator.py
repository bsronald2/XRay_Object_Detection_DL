from keras.preprocessing.image import ImageDataGenerator
import numpy as np


class DataGenerator:

    def __init__(self):
        self.processed_dir = 'data/processed'

    def generate_data_K(self, ds, save_to_dir=None, prefix='test'):
        if save_to_dir is None:
            save_to_dir = self.processed_dir
        else:
            None

        img_gen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            featurewise_center=True,
            samplewise_center=True,
            featurewise_std_normalization=True,
            samplewise_std_normalization=True,
            horizontal_flip=True,
        )

        for img, _ in ds.as_numpy_iterator():
            img_flow = img_gen.flow(img, batch_size=32, seed=1, save_to_dir=str(save_to_dir), save_prefix=prefix)
            dummy = [next(img_flow)[0].astype(np.uint8) for i in range(5)]
