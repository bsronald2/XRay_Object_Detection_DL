import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.Unet import Unet
from src.config import dim, n_classes, n_filters, labels, model_name, ann_file_name, multiply_by, batch_size
from src.data.GDXrayDataGenerator import GDXrayDataGenerator
from src.models.metrics import (dice, dice_coef, bce_dice_loss)
from src.models.iou_metric import iou
from src.utils import delete_file, create_random_list_of_size, save_model_history
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


@click.command()
@click.argument('input_img_path', type=click.Path(exists=True))
@click.argument('ann_path', type=click.Path(exists=True))
@click.option('--model-type', type=click.Choice(['unet'], case_sensitive=False))
@click.option('--metric', type=click.Choice(['acc', 'dice', 'iou'], case_sensitive=False))
def main(input_img_path, ann_path, model_type, metric):
    """
    Runs training process and save model.
    """
    logger = logging.getLogger(__name__)

    # collect images path
    input_img_dir = Path(input_img_path)
    imgs_path_train = sorted([i.absolute() for i in (input_img_dir / 'train').glob("*.png") if i.is_file()])
    imgs_path_val = sorted([i.absolute() for i in (input_img_dir / 'val').glob("*.png") if i.is_file()])
    imgs_path_test = sorted([i.absolute() for i in (input_img_dir / 'test').glob("*.png") if i.is_file()])

    # Annotations path
    ann_path_dir = Path(ann_path)
    ann_train_path = ann_path_dir / 'train' / ann_file_name
    ann_val_path = ann_path_dir / 'val' / ann_file_name
    ann_test_path = ann_path_dir / 'test' / ann_file_name

    # Add randomly more paths
    imgs_path_train = create_random_list_of_size(imgs_path_train, len(imgs_path_train) * multiply_by)
    imgs_path_val = create_random_list_of_size(imgs_path_val, len(imgs_path_val) * multiply_by)
    imgs_path_test = create_random_list_of_size(imgs_path_test, len(imgs_path_test) * multiply_by)

    # Generate Data on the fly for train and validation
    data_generator_train = GDXrayDataGenerator(imgs_path_train, ann_train_path, labels, n_classes,
                                               batch_size=batch_size,
                                               dim=dim)
    data_generator_val = GDXrayDataGenerator(imgs_path_val, ann_val_path, labels, n_classes, batch_size=batch_size,
                                             dim=dim)
    data_generator_test = GDXrayDataGenerator(imgs_path_test, ann_test_path, labels, n_classes, batch_size=batch_size,
                                              dim=dim)

    # Model Path
    model_path = Path('models') / (model_name % model_type)
    delete_file(model_path)

    # Set-up model selected
    model = Unet(dim, n_classes, n_filters)
    conf, call_backs = get_model_configures(metric, str(model_path))
    model.build_model(**conf)

    # Fit the model TODO add timer
    history = model.fit(
        x=data_generator_train,
        steps_per_epoch=len(data_generator_train),
        validation_data=data_generator_val,
        epochs=50,
        verbose=1,
        callbacks=call_backs
    )

    # Evaluate model
    result = model.evaluate(data_generator_test)
    print(result)
    print(dict(zip(model.metrics_names, result)))
    model.save_model(str(model_path))

    # Save History
    save_history(metric, history)


def get_model_configures(metric, model_path):
    if metric == 'acc':
        conf = {'optimizer': Adam(), 'loss': 'categorical_crossentropy', 'metrics': ["accuracy"]}
        call_backs = [ModelCheckpoint(model_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True,
                                      save_weights_only=True),
                      EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='min'),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, mode='min', cooldown=5, min_lr=1e-5)]
    elif metric == 'dice':
        conf = {'optimizer': Adam(), 'loss': 'categorical_crossentropy', 'metrics': [dice_coef]}
        call_backs = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, epsilon=1e-4,
                                        mode='min'),
                      ModelCheckpoint(model_path, monitor='val_loss', verbose=1, mode='min',
                                      save_best_only=True, save_weights_only=True)]
    elif metric == 'iou':
        conf = {'optimizer': Adam(), 'loss': 'categorical_crossentropy', 'metrics': [iou]}
        call_backs = [
            EarlyStopping(monitor='val_iou', patience=15, verbose=1, min_delta=1e-4, mode='max'),
            ReduceLROnPlateau(monitor='val_iou', factor=0.2, patience=5, verbose=1, epsilon=1e-4,
                              mode='max'),
            ModelCheckpoint(model_path, monitor='val_iou', verbose=1, mode='max',
                            save_best_only=True, save_weights_only=True)]

    else:
        raise ValueError(f"Unrecognized parameter: {metric}")

    return conf, call_backs


def save_history(metric, history):
    if metric == 'acc':
        acc_dic = {'y': history.history['accuracy'], 'X': history.history['val_accuracy'], 'title': 'Accuracy',
                   'ylabel': 'accuracy', 'xlabel': 'epoch', 'legend': ['train', 'val']}
        loss_dic = {'y': history.history['loss'], 'X': history.history['val_loss'], 'title': 'model loss',
                    'ylabel': 'loss', 'xlabel': 'epoch', 'legend': ['train', 'val']}
        save_model_history(acc_dic)
        save_model_history(loss_dic)
    elif metric == 'dice':
        dice_dic = {'y': history.history['dice_coef'], 'X': history.history['val_dice_coef'], 'title': 'Dice',
                   'ylabel': 'dice', 'xlabel': 'epoch', 'legend': ['train', 'val']}
        loss_dic = {'y': history.history['loss'], 'X': history.history['val_loss'], 'title': 'model loss',
                    'ylabel': 'loss', 'xlabel': 'epoch', 'legend': ['train', 'val']}
        save_model_history(dice_dic)
        save_model_history(loss_dic)
    elif metric == 'iou':
        dice_dic = {'y': history.history['iou'], 'X': history.history['val_iou'], 'title': 'IoU',
                    'ylabel': 'dice', 'xlabel': 'epoch', 'legend': ['train', 'val']}
        loss_dic = {'y': history.history['loss'], 'X': history.history['val_loss'], 'title': 'model loss',
                    'ylabel': 'loss', 'xlabel': 'epoch', 'legend': ['train', 'val']}
        save_model_history(dice_dic)
        save_model_history(loss_dic)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
