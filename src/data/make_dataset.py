# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import split_folders
from dotenv import find_dotenv, load_dotenv
from src.data.GDXray import GDXray


@click.command()
@click.argument('input_img_path', type=click.Path(exists=True))
@click.argument('output_img_path', type=click.Path())
@click.option('--data-aug', is_flag=True, help="Create a synthetic data-set")
def main(input_img_path, output_img_path, data_aug):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    input_data_dir = Path(input_img_path)
    output_data_dir = Path(output_img_path)
    # TODO add condition to split dataset into train, val, test
    # split_folders.ratio(raw_imagepath, output=output_filepath, seed=1337, ratio=(.8, .1, .1))  # default values
    if data_aug is True:
        logger.info('Create a synthetic data-set.')
        gdx = GDXray()
        gdx.pre_process(input_dir=input_data_dir, output_dir=output_data_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
