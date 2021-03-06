import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.contours import predict_contours_batch
from src.utils import create_dir, create_random_list_of_size, save_iou_th
from src.models.Unet import Unet
from src.config import dim, n_classes, n_filters, ann_file_name, labels, batch_size
from src.data.GDXrayDataGenerator import GDXrayDataGenerator
from src.models.mask_utils import Mask
from src.models.metrics import metric_batch
import numpy as np


@click.command()
@click.argument('input_img_path', type=click.Path(exists=True))
@click.argument('ann_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('output_pred_path', type=click.Path())
@click.option('--model-type', type=click.Choice(['unet', 'contours'], case_sensitive=False))
@click.option('--is-batch', is_flag=True)
def main(input_img_path, ann_path, output_pred_path, model_path, is_batch, model_type):
    """
    Predict Images.
    """
    logger = logging.getLogger(__name__)
    # Create Directory if doesn't exits otherwise remove items inside it.
    create_dir(Path(output_pred_path))

    # collect images path
    input_img_dir = Path(input_img_path)
    imgs_path_test = sorted([i.absolute() for i in (input_img_dir / 'test').glob("*.png") if i.is_file()])
    # Annotation path
    ann_path_dir = Path(ann_path)
    ann_test_path = ann_path_dir / 'test' / ann_file_name

    imgs_path_test = create_random_list_of_size(imgs_path_test, len(imgs_path_test) * 3)
    mask = Mask(output_pred_path)

#    calculate_iou_metric(model, data_generator_test, mask)
    if model_type == 'unet':
        # Load model pre-trained
        data_generator_test = GDXrayDataGenerator(imgs_path_test, ann_test_path, labels, n_classes,
                                                  batch_size=batch_size, dim=dim)
        model = Unet(dim, n_classes, n_filters=n_filters, pretrained_weights=model_path)
        predict_unet(model, data_generator_test, mask)
    elif model_type == 'contours':
        data_generator_test = GDXrayDataGenerator(imgs_path_test, ann_test_path, labels, n_classes,
                                                  batch_size=batch_size, dim=dim, task='binary')
        predict_contours_batch(data_generator_test, mask)


def predict_unet(model, data_generator_test, mask):
    metrics = list()
    iou = list()
    for X, y_exp in data_generator_test.get_iter():
        y_pred = model.predict(X)
        metrics.append(metric_batch(y_exp, y_pred, calc_metric='dice'))
        iou.append(metric_batch(y_exp, y_pred, calc_metric='iou'))
        # Save predictions
        mask.blending_batch(X, y_pred)

    dice_value = np.mean(metrics)
    iou_metric = np.mean(iou)
    print("Dice Metric:", dice_value)
    print("IoU Metric:", iou_metric)


def calculate_iou_metric(model, data_generator_test, mask):
    metric = list()
    th_steps = 25
    thresholds = np.linspace(0, 1, th_steps)
    for X, y_exp in data_generator_test.get_iter():
        y_pred = model.predict(X)
        metric.extend(np.array([metric_batch(y_exp, np.int32(y_pred > threshold)) for threshold in
                                thresholds]))
        # Save predictions
        mask.blending_batch(X, y_pred)

        # Retrieve best iou and threshold
    metric = np.array(metric).reshape(3, th_steps)
    metric_mean = metric.mean(axis=0)
    best_threshold_index = np.argmax(metric_mean)
    best_iou = metric_mean[best_threshold_index]
    print('Best IoU mean:', best_iou)
    best_threshold = thresholds[best_threshold_index]
    print('Best Threshold:', best_threshold)
    save_iou_th(thresholds, metric_mean, best_threshold, best_iou)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
