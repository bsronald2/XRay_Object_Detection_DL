from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from src.data import data_utils
import random
# To Create Synthetic data-set
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.50))
    ),
    # Strengthen or weaken the contrast in each images.
    iaa.LinearContrast((0.75, 1)),

    # Add gaussian noise.
    # For 30% of all images, we sample the noise once per pixel.
    # For the other 30% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.3),

    # Apply affine transformations to each images.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order


class DataGenerator:
    """
     Methods to create synthetic dataset to use for instance in a classification task.
    """
    def __init__(self):
        self.processed_dir = 'data/processed'

    def generate_data_K(self, ds, save_to_dir=None, prefix='test', data_gen=None):
        """ Generates augmented images using Keras ImageDataGenerator class.
            The results are persisted in disc."""
        if save_to_dir is None:
            save_to_dir = self.processed_dir
        else:
            data_utils.create_dir(save_to_dir)

        img_gen = ImageDataGenerator(**data_gen)
        # Create by default 5 new augmented pictures by each original images.
        for img, _ in ds.as_numpy_iterator():
            img_flow = img_gen.flow(img, batch_size=32, save_to_dir=str(save_to_dir), save_prefix=prefix)
            [next(img_flow)[0].astype(np.uint8) for i in range(5)]

    def generate_data_I(self, ds, save_to_dir=None, prefix='test'):
        """
        Generates augmented images using ImgAug library class.
        The results are persisted in disc.
        """
        data_utils.create_dir(save_to_dir)
        for imgs_batch, _ in ds.as_numpy_iterator():
            batches = UnnormalizedBatch(images=(imgs_batch*255).astype(np.uint8))
            images_aug = [next(seq.augment_batches(batches, background=True)).images_aug for i in range(5)]
            [imageio.imwrite("%s/%s_%d_%d.png" % (str(save_to_dir), prefix, i, random.randint(0, 1000),), ia_j)
             for i, images in enumerate(images_aug) for ia_j in images]
