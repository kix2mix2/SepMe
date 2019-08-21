import click
import os
import random


import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id
from SepMe.utils.workflow_utils import _get_or_run
from SepMe import logger

@click.command()
@click.option("--exp_name", default='SepMe', type=str)
def workflow(exp_name):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.

    #dir = os.path.dirname(os.path.abspath(__file__)).split('/SepMe')[0]
    ddir = os.path.dirname(os.path.abspath(__file__))
    logger.info(ddir)
    logger.info('------------------\n')

    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name="DSC_wf") as active_run:

        logger.log_metric('kii', 1)
        os.environ['SPARK_CONF_DIR'] = os.path.abspath('.')
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        filter_study_data = _get_or_run(ddir, "filter_study_data", {'path': '/Users/morarica/Developer/SepMe/data/RESULTS_EUROVIS2015.csv'}, git_commit)

        print(filter_study_data)
        dsc = _get_or_run(ddir, "dsc", {}, git_commit)

        print('hello')


if __name__ == '__main__':
    workflow()
