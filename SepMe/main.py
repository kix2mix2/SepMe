import click
import mlflow
import pandas as pd
import yaml
from mlflow.utils import mlflow_tags

from SepMe import logger
from SepMe.utils.graph_utils import calculate_graphs, calculate_purities
from SepMe.utils.workflow_utils import load_yaml
import pickle


@click.command()
@click.option("--config-path", default="SepMe/configs/baby_config.yaml")
def workflow(config_path):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with open(config_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    logger.info(config)
    mlflow.set_experiment(config['experiment_name'])
    data_dict = {}

    # read in list of datasets
    data_df = pd.read_csv(config['data_path'])

    # for each dataset
    for i, file in enumerate(data_df.fileName):
        file_name = (
            config['folder_path']
            + file.split(".csv")[0]
            + "_cls"
            + str(data_df.classNum[i])
            + ".csv"
        )
        code_name = (file.split(".csv")[0]
            + "_cls"
            + str(data_df.classNum[i]))
        try:
            df = pd.read_csv(file_name, names=["x", "y", "class"])
        except FileNotFoundError:
            df = None
            logger.info("File '" + file + "' does not exist.")
            # data_dict[file_name] = 'n/a'
            continue

        if df is not None:
            with mlflow.start_run(run_name=code_name) as active_run:
                data_dict[file] = {}
                logger.info("Runtime: " + code_name)
                logger.log_param('df_size', len(df))

                nx_dict = calculate_graphs(df, config['graph_types']) # calculate all graphs

                with open(code_name + '_graphs.pickle', 'wb') as handle:
                    pickle.dump(nx_dict, handle, protocol = pickle.HIGHEST_PROTOCOL)

                #logger.log_artifact(file + '_graphs.pickle')

                for graph in nx_dict.keys():
                    logger.info('Calculating purity for ' + graph)
                    data_dict[file][graph] = {}
                    if type(nx_dict[graph]) is dict:
                        for subgraph in nx_dict[graph].keys():
                            data_dict[file][graph][subgraph] = calculate_purities(df, nx_dict[graph][subgraph], config['purities'])
                    else:
                        data_dict[file][graph] = calculate_purities(df, nx_dict[graph], config['purities'])

                logger.info('This is the collected data: ')
                logger.info(data_dict)

        #logger.info(data_dict)


        # for each graph calculate all purities
        # save each result in a per dataset dict


if __name__ == "__main__":
    workflow()
