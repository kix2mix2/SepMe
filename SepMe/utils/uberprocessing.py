import json

# import mlflow
import pandas as pd
import ray
from flatten_dict import flatten

from SepMe.graph import calculate_graphs, calculate_purities
from SepMe.utils.workflow_utils import timeit, underscore_reducer


@ray.remote
def process_dataset(file, config, i, lf, dict_dir, save=False):
    # logger = get_logger("SepMe_"+str(i), "../sepme_" + str(i) +'.log')
    print("------------")
    print("---- Processing file {}/{}. Name: {}".format(i, lf, file.split(".csv")[0]))

    data_dict = {}
    graph_dir = config["graph_path"] + config["experiment_name"] + "/"
    code_name = file.split(".csv")[0]

    try:
        df = pd.read_csv(config["folder_path"] + file)
        if len(df) < 10:
            return {}
    except FileNotFoundError:
        print("File {} doesn't exist".format(file))
        df = None
        return {}

    # process file
    if df is not None:
        # with mlflow.start_run(run_name=code_name):
        print("Runtime: " + code_name)
        print("df_size", len(df))

        data_dict[code_name] = {}

        graph_path = graph_dir + code_name + "_graphs.pickle"
        nx_dict = calculate_graphs(graph_path, df, config["graph_types"])

        # logger.info("Calculating purities..")
        for graph in nx_dict.keys():
            data_dict[code_name][graph] = {}
            if type(nx_dict[graph]) is dict:
                for subgraph in nx_dict[graph].keys():
                    data_dict[code_name][graph][subgraph] = calculate_purities(
                        df, nx_dict[graph][subgraph], config["purities"],
                    )
            else:
                data_dict[code_name][graph] = calculate_purities(
                    df, nx_dict[graph], config["purities"]
                )

    if save:
        with open(dict_dir + file.split(".csv")[0] + ".json", "w") as fp:
            json.dump(data_dict, fp)

    return data_dict


@timeit
def save_progress(data_dict, results_path):
    data_new = {}
    for file in data_dict.keys():
        data_new[file] = flatten(data_dict[file], reducer=underscore_reducer)
    results_df = pd.DataFrame.from_dict(data_new, orient="index")

    # results_all = pd.read_csv(results_path, index_col = 0)
    # results_all = results_all.append(results_df, sort = False)
    # results_all.to_csv(results_path)
    print(results_path)
    results_df.to_csv(results_path)

    return {}
