import json
import os
import click
import mlflow
import pandas as pd
import yaml
from flatten_dict import flatten
from SepMe import logger
from SepMe.utils.graph_utils import calculate_graphs, calculate_purities
from SepMe.utils.workflow_utils import underscore_reducer, timeit, timeit_print


@timeit
def save_progress(data_dict, results_path, config, i):
    logger.info("This is the collected data: ")
    logger.info(data_dict)


    # resave results
    # with open(results_path + config["experiment_name"] + '_' + str(i) + "_data.json", "w") as fp:
    #     json.dump(data_dict, fp)

    # flatten dict and save to dataframe
    data_new = {}
    for file in data_dict.keys():
        data_new[file] = flatten(data_dict[file], reducer = underscore_reducer)
    results_df = pd.DataFrame.from_dict(data_new, orient = "index")


    if os.path.exists(results_path):
        results_all = pd.read_csv(results_path, index_col=0)
        results_all = results_all.append(results_df, sort=False)
        results_all.to_csv(results_path)

    else:
        results_df.to_csv(results_path)

    # return empty dict to free memory.
    return {}

@timeit_print
@click.command()
@click.option("--config-path", default = "SepMe/configs/baby_config.yaml")
@click.option("--save", default = 10)
def workflow(config_path, save):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    logger.info(config)
    mlflow.set_experiment(config["experiment_name"])
    data_dict = {}

    # read in list of input datasets
    data_df = pd.read_csv(config["data_path"])

    # read in pre-existent results
    results_path = config["results_path"] + config["experiment_name"] + ".csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, index_col=0)
    else:
        results_df = None

    # make a directory to save graphs
    graph_dir = config["graph_path"] + config["experiment_name"] + "/"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # for each dataset
    for i, file in enumerate(data_df.fileName):
        code_name = file.split(".csv")[0] + "_cls" + str(data_df.classNum[i])
        file_name = config["folder_path"] + code_name + ".csv"

        # check if file was already processed
        if (results_df is not None) and (code_name in results_df.index):
            logger.info(
                "File '"
                + code_name
                + "' was previously processed. Skipping..."
            )
            continue

        # check is file exists
        try:
            df = pd.read_csv(file_name, names = ["x", "y", "class"])
        except FileNotFoundError:
            df = None
            logger.info("File '" + file + "' does not exist.")
            # data_dict[file_name] = 'n/a'
            continue

        # process file
        if df is not None:
            with mlflow.start_run(run_name = code_name) as active_run:
                logger.info("Runtime: " + code_name)
                logger.log_param("df_size", len(df))

                data_dict[code_name] = {}

                graph_path = graph_dir + code_name + "_graphs.pickle"
                nx_dict = calculate_graphs(graph_path, df, config["graph_types"])

                logger.info("Calculating purities..")
                for graph in nx_dict.keys():
                    data_dict[code_name][graph] = {}
                    if type(nx_dict[graph]) is dict:
                        for subgraph in nx_dict[graph].keys():
                            data_dict[code_name][graph][subgraph] = calculate_purities(df, nx_dict[graph][subgraph],
                                                                                       config["purities"],
                                                                                       )
                    else:
                        data_dict[code_name][graph] = calculate_purities(df, nx_dict[graph], config["purities"])

                # every save files processed save the results
                if i % save == 0:
                    logger.info('Saving progress...')
                    data_dict = save_progress(data_dict, results_path, config, i)

        print('-------------------------------------')


if __name__ == "__main__":
    workflow()
