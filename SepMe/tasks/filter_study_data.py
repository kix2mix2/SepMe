"""
Downloads the MovieLens dataset and saves it as an artifact
"""

from __future__ import print_function

import click
import mlflow
import pandas as pd
import numpy as np

from SepMe import logger


@click.command(
    help="Downloads the MovieLens dataset and saves it as an mlflow artifact "
    " called 'ratings-csv-dir'."
)
@click.option("--path", default="data/RESULTS_EUROVIS2015.csv")
@click.option("--col_a", default="scoreA", type=str)
@click.option("--col_m", default="scoreM", type=str)
@click.option("--filter_out", default=[0, 3])
@click.option("--max_diff", default=0, type=int)
def filter_study_data(path, col_a, col_m, filter_out, max_diff):
    with mlflow.start_run(run_name="filter_study_data") as mlrun:
        logger.log_param("path", path)
        logger.log_param("filter_out", filter_out)
        logger.log_param("max_diff", max_diff)

        df = pd.read_csv(path)

        df = df[
            (~df[col_a].isin(filter_out))
            & (~df[col_m].isin(filter_out))
            & (np.abs(df[col_a] - df[col_m]) <= max_diff)
        ]
        logger.log_metric(
            "df_rows",
            len(
                df[
                    (~df[col_a].isin(filter_out))
                    & (~df[col_m].isin(filter_out))
                    & (np.abs(df[col_a] - df[col_m]) <= max_diff)
                ]
            ),
        )

        mlrun.random = df


if __name__ == "__main__":
    filter_study_data()
