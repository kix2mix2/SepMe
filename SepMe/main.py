import os
import time

import click
import mlflow
import pandas as pd
import yaml
from flatten_dict import flatten
from SepMe import logger
from SepMe.utils.graph_utils import calculate_graphs, calculate_purities
from SepMe.utils.workflow_utils import underscore_reducer, timeit
import ray
import psutil
import pickle

from SepMe.utils.logger import get_logger

@timeit
def save_progress(data_dict, results_path):
    data_new = {}
    for file in data_dict.keys():
        data_new[file] = flatten(data_dict[file], reducer = underscore_reducer)
    results_df = pd.DataFrame.from_dict(data_new, orient = "index")
    results_df.to_csv(results_path)
    # if False:
    #     results_all = pd.read_csv(results_path, index_col = 0)
    #     results_all = results_all.append(results_df, sort = False)
    #     results_all.to_csv(results_path)
    #
    # else:
    #     results_df.to_csv(results_path)

    # return empty dict to free memory.
    return {}

@ray.remote
def process_dataset(i, file, config, class_num):
    #logger = get_logger("SepMe_"+str(i), "../sepme_" + str(i) +'.log')
    print('----------------- ' + str(i) + ' -------------------')
    data_dict = {}
    graph_dir = config["graph_path"] + config["experiment_name"] + "/"
    code_name = file.split(".csv")[0] + "_cls" + str(class_num[i])
    file_name = config["folder_path"] + code_name + ".csv"

    try:
        df = pd.read_csv(file_name, names = ["x", "y", "class"])
    except FileNotFoundError:
        df = None
        #logger.info("File '" + file + "' does not exist.")
        # data_dict[file_name] = 'n/a'
        return

    # process file
    if df is not None:
        with mlflow.start_run(run_name = code_name) as active_run:
            #logger.info("Runtime: " + code_name)
            #logger.log_param("df_size", len(df))

            data_dict[code_name] = {}

            graph_path = graph_dir + code_name + "_graphs.pickle"
            nx_dict = calculate_graphs(graph_path, df, config["graph_types"])

            #logger.info("Calculating purities..")
            for graph in nx_dict.keys():
                data_dict[code_name][graph] = {}
                if type(nx_dict[graph]) is dict:
                    for subgraph in nx_dict[graph].keys():
                        data_dict[code_name][graph][subgraph] = calculate_purities(df, nx_dict[graph][subgraph],
                                                                                   config["purities"],
                                                                                   )
                else:
                    data_dict[code_name][graph] = calculate_purities(df, nx_dict[graph], config["purities"])


    return data_dict


@click.command()
@click.option("--config-path", default = "SepMe/configs/config.yaml")
@click.option("--save", default = 5)
def workflow(config_path, save):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    ray.init(num_cpus=psutil.cpu_count())
    time.sleep(2.0)
    start_time = time.time()

    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return

    logger.info(config)
    logger.info('Number of processors: '+ str(psutil.cpu_count()))
    mlflow.set_experiment(config["experiment_name"])

    # read in list of input datasets
    data_df = pd.read_csv(config["data_path"])
    #data_df = data_df[:20]

    # make a directory to save graphs
    graph_dir = config["graph_path"] + config["experiment_name"] + "/"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    class_num = list(data_df.classNum)
    file_name = list(data_df.fileName)

    results = []
    for i, file in enumerate(file_name):
        results.append(process_dataset.remote(i, file, config, class_num))

    results = ray.get(results)

    res_dict={}
    for res in results:
        res_dict.update(res)
    results_path = config["results_path"] + config["experiment_name"] + ".csv"
    save_progress(res_dict, results_path)

    end_time = time.time()
    duration = end_time - start_time
    logger.info("Success! The workflow took {} seconds.".format(duration))


if __name__ == "__main__":
    print('hello')

    workflow()

