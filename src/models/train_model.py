import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.models.Unet import Unet
from src.config import dim, n_classes, n_filters, labels, model_name, ann_file_name, multiply_by, batch_size
from src.data.GDXrayDataGenerator import GDXrayDataGenerator
from src.utils import delete_file, create_random_list_of_size, save_model_history


@click.command()
@click.argument('input_img_path', type=click.Path(exists=True))
@click.argument('ann_path', type=click.Path(exists=True))
@click.option('--model-type', type=click.Choice(['unet'], case_sensitive=False))
def main(input_img_path, ann_path, model_type):
    """
    Runs training process and save model.
    """
    logger = logging.getLogger(__name__)

    # collect images path and annotations path
    input_img_dir = Path(input_img_path)
    imgs_path_train = sorted([i.absolute() for i in (input_img_dir / 'train').glob("*.png") if i.is_file()])
    imgs_path_val = sorted([i.absolute() for i in (input_img_dir / 'val').glob("*.png") if i.is_file()])
    ann_path_dir = Path(ann_path)
    ann_train_path = ann_path_dir / 'train' / ann_file_name
    ann_val_path = ann_path_dir / 'val' / ann_file_name

    imgs_path_train = create_random_list_of_size(imgs_path_train, len(imgs_path_train) * multiply_by)
    imgs_path_val = create_random_list_of_size(imgs_path_val, len(imgs_path_val) * multiply_by)

    # Generate Data on the fly for train and validation
    data_generator_train = GDXrayDataGenerator(imgs_path_train,
                                               ann_train_path, labels, n_classes, batch_size=batch_size, dim=dim)
    data_generator_val = GDXrayDataGenerator(imgs_path_val,
                                             ann_val_path, labels, n_classes, batch_size=batch_size, dim=dim)

    # Model Path
    model_path = Path('models') / (model_name % model_type)
    delete_file(model_path)

    # Set-up model selected TODO add if conditional to select model type
    model = Unet(dim, n_classes, n_filters)
    model.build()
    model_checkpoint = model.checkpoint(str(model_path))
    model_early_stop = model.early_stopping()
    # Fit the model
    history = model.fit(
        x=data_generator_train,
        steps_per_epoch=len(data_generator_train),
        validation_data=data_generator_val,
        epochs=2,
        verbose=1,
        callbacks=[model_early_stop, model_checkpoint]
    )

    model.save_model(str(model_path))

    # Save History
    acc_dic = {'y': history.history['accuracy'], 'X': history.history['val_accuracy'], 'title': 'model accuracy',
               'ylabel': 'accuracy', 'xlabel': 'epoch', 'legend': ['train', 'val']}
    loss_dic = {'y': history.history['loss'], 'X': history.history['val_loss'], 'title': 'model loss',
               'ylabel': 'loss', 'xlabel': 'epoch', 'legend': ['train', 'val']}
    save_model_history(acc_dic)
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
