"""
Downloads the MovieLens dataset and saves it as an artifact
"""


from __future__ import print_function
from SepMe import logger


import requests
import tempfile
import os
import zipfile
import mlflow
import click


@click.command(help="Downloads the MovieLens dataset and saves it as an mlflow artifact "
                    " called 'ratings-csv-dir'.")
@click.option("--path", default="http://files.grouplens.org/datasets/movielens/ml-20m.zip")
def load_raw_data(path):
    with mlflow.start_run() as mlrun:
        mlflow.log_metric("foo", 3)
        logger.log_metric('ki', 1)


if __name__ == '__main__':
    load_raw_data()