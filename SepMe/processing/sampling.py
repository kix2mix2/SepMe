import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile


def sample_data(df, y_col, amount):
    X_train, X_test, y_train, y_test = train_test_split(
        df, y_col, test_size=amount, random_state=21
    )
    return y_test.index


def create_sample_from_index(
    df,
    index,
    fig_folder="../../data/orig_data/figures/reduced_data/",
    csv_folder="../../data/orig_data/input_data/Reduced_orig_data/reduced_clean/",
    save_folder=" ../../data/mturk_samples/sample_test/",
):
    # create end destination directory
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(save_folder + "data/"):
        os.makedirs(save_folder + "data/")

    if not os.path.exists(save_folder + "figures/"):
        os.makedirs(save_folder + "figures/")

    tuples = []

    # copy files to mturk folder
    pngs = [row + ".png" for i, row in df.loc[index, "index"].items()]
    for png in pngs:
        if os.path.exists(fig_folder + png):
            copyfile(fig_folder + png, save_folder + "figures/" + png)
        else:
            print("File {} foes not exist".format(png))

    csvs = [row + ".csv" for i, row in df.loc[index, "index"].items()]
    for i, csv in enumerate(csvs):
        if os.path.exists(csv_folder + csv):
            copyfile(csv_folder + csv, save_folder + "data/" + csv)

            df = pd.read_csv(csv_folder + "sample/" + csv)
            tuples.append(
                [
                    "https://scatterplots.s3.eu-central-1.amazonaws.com/"
                    + pngs[i].split(".png")[0]
                    + ".png",
                    int(len(set(df["class"]))),
                ]
            )
        else:
            print("File {} foes not exist".format(csv))

    # create batch file for this run
    tuples = np.array(tuples)
    ddf = pd.DataFrame({"image_url": tuples[:, 0], "i": tuples[:, 1]})
    ddf.to_csv(save_folder + "batch_index.csv")


def get_one_df(
    ii=5,
    data_path="../data/RESULTS_EUROVIS2015.csv",
    folder_path="../data/EUROVIS_new/",
):
    df = pd.read_csv(data_path)
    for i, file in enumerate(df.fileName):
        if i != ii:
            continue

        file_name = (
            folder_path + file.split(".csv")[0] + "_cls" + str(df.classNum[i]) + ".csv"
        )
        try:
            sample_df = pd.read_csv(file_name, names=["x", "y", "class"])
        except FileNotFoundError:
            print("File '" + file + "' does not exist.")

    # sample_df.head()
    return sample_df
