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

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score


def calculate_dsc(sample_df, class_names):
    centroids = {}
    centroids_list = []
    for class_value in set(sample_df['class']):
        centroids[class_value] = tuple(np.mean(sample_df[sample_df['class'] == class_value])[0:2])
        centroids_list.append(tuple(np.mean(sample_df[sample_df['class'] == class_value])[0:2]))

    a = np.argmin(cdist(sample_df[['x', 'y']], centroids_list), axis=1)

    if class_names==True:
        names = [list(centroids.keys())[aa] for aa in a]
    else:
        names = a
    sample_df['cd'] = names
    dsc = accuracy_score(sample_df['cd'], sample_df['class'])
    return dsc


@click.command(help="Runs binary class experiment.")
@click.option("--data-path", default="data/RESULTS_EUROVIS2015.csv")
@click.option("--folder-path", default="data/EUROVIS_new/")
@click.option("--class_names", default=False)
def dsc(data_path, folder_path, class_names):
    with mlflow.start_run(run_name="filter_study_data") as mlrun:
        logger.log_param('data_path', data_path)
        logger.log_param('folder_path', folder_path)

        df = pd.read_csv(data_path)
        dsc_list = []

        for i, file in enumerate(df.fileName):
            file_name = folder_path + file.split('.csv')[0] + '_cls' + str(df.classNum[i]) + '.csv'
            try:
                sample_df = pd.read_csv(file_name, names=['x', 'y', 'class'])
                dsc = calculate_dsc(sample_df, class_names)
                dsc_list.append(dsc*100)
                #logger.log_metric('dsc_'+file+'_cls'+str(df.classNum[i]), dsc)


            except FileNotFoundError:
                logger.info('File \'' + file + '\' does not exist.')
                dsc_list.append(0)

        df['DSC_mlflow'] = dsc_list
        df[['fileName', 'classNum', 'DSC_mlflow', 'DSC']].to_csv('data/results/df.csv', index=False)
        logger.log_artifact('data/results/df.csv')
        logger.log_metric('dsc_mean', np.mean(dsc_list))
        logger.log_metric('dsc_std', np.std(dsc_list))



if __name__ == '__main__':
    dsc()
