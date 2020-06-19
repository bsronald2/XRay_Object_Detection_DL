import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import create_dir
from src.models.Unet import Unet
from src.config import dim, n_classes
from src.data.GDXray import GDXray
import cv2
import numpy as np
import imageio
import imgaug as ia


@click.command()
@click.argument('input_img_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('output_pred_path', type=click.Path())
@click.option('--model-type', type=click.Choice(['unet'], case_sensitive=False))
@click.option('--is-batch', is_flag=True)
def main(input_img_path, output_pred_path, model_path, is_batch, model_type):
    """
    Predict Images.
    """
    logger = logging.getLogger(__name__)
    # Create Directory if doesn't exits otherwise remove items inside it.
    create_dir(Path(output_pred_path))
    # Load model pre-trained
    model = Unet(dim, n_classes, n_filters=16, pretrained_weights=model_path)
    # collect images path
    gdx_ray = GDXray(input_img_path, train_val_ds=False)
    if is_batch:
        # TODO save batch predictions
        test_ds = gdx_ray.load_dataset()
        pred = model.predict(test_ds)
    else:
        # Predict mask for image
        img = gdx_ray.load_img()
        img = np.expand_dims(img, axis=0)
        pred_mask = model.predict(img)[0]

        # merge image and mask
        mask2 = cv2.addWeighted(img.squeeze(), 1.0, (pred_mask.squeeze() * 255.), 5.0, 0)

        # save image
        input_path = Path(input_img_path)
        file_name = Path(output_pred_path) / (input_path.name.split('.')[0] + '_predicted' + '.png')
        imageio.imwrite(str(file_name), mask2 * 255.)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()