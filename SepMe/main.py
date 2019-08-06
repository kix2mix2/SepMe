"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.

See README.rst for more details.
"""

import click
import os


import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id
from SepMe.utils.workflow_utils import _get_or_run
from SepMe import logger

@click.command()
@click.option("--als-max-iter", default=10, type=int)
@click.option("--keras-hidden-units", default=20, type=int)
@click.option("--max-row-limit", default=100000, type=int)
def workflow(als_max_iter, keras_hidden_units, max_row_limit):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.

    #dir = os.path.dirname(os.path.abspath(__file__)).split('/SepMe')[0]
    dir = os.path.dirname(os.path.abspath(__file__))
    print(dir)
    print('------------------\n')


    with mlflow.start_run() as active_run:

        logger.log_metric('kii', 1)

        os.environ['SPARK_CONF_DIR'] = os.path.abspath('.')
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        load_raw_data_run = _get_or_run(dir, "load_raw_data", {}, git_commit)




if __name__ == '__main__':
    workflow()
