import ray
import os
import time
import click

# import mlflow
import yaml
from SepMe import logger
from SepMe.utils.uberprocessing import process_dataset, save_progress
import psutil


@click.command()
@click.option("--config-path", default="SepMe/configs/baby_config.yaml")
@click.option("--save", default=5)
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
    logger.info("Number of processors: " + str(psutil.cpu_count() - 2))
    # mlflow.set_experiment(config["experiment_name"])

    # make a directory to save graphs
    graph_dir = config["graph_path"] + config["experiment_name"] + "/"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    dict_dir = config["dict_path"] + config["experiment_name"] + "/"
    if not os.path.exists(dict_dir):
        os.makedirs(dict_dir)

    files = os.listdir(config["folder_path"])[:5]
    # files = ['boston_umap2-mds2.csv']
    results = []

    lf = len(files)
    for i, file in enumerate(files):

        if file.endswith(".csv"):
            results.append(
                process_dataset.remote(file, config, i, lf, dict_dir, save=True)
            )
        else:
            continue

    results = ray.get(results)

    res_dict = {}
    for res in results:
        res_dict.update(res)
    results_path = config["results_path"] + config["experiment_name"] + ".csv"
    save_progress(res_dict, results_path)

    end_time = time.time()
    duration = end_time - start_time
    logger.info("Success! The workflow took {} seconds.".format(duration))


if __name__ == "__main__":
    print("Hello World!")

    workflow()
