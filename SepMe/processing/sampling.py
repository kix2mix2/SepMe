import os

import boto3
import botocore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile


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


def get_samples(df, y_col, amount):
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
    print("ok")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(save_folder + "data/"):
        print("ok")
        os.makedirs(save_folder + "data/")

    if not os.path.exists(save_folder + "figures/"):
        print("ok")
        os.makedirs(save_folder + "figures/")

    # copy files to mturk folder
    pngs = [row + ".png" for i, row in df.loc[index, "index"].items()]
    for png in pngs:
        if os.path.exists(fig_folder + png):
            copyfile(fig_folder + png, save_folder + "figures/" + png)
        else:
            print("----")
            print("File {} foes not exist".format(png))

    csvs = [row + "_1-2.csv" for i, row in df.loc[index, "index"].items()]
    for i, csv in enumerate(csvs):
        if os.path.exists(csv_folder + csv):
            copyfile(csv_folder + csv, save_folder + "data/" + csv)
        else:
            print("File {} foes not exist".format(csv))
            print("----")


def create_s3_batchfile(
    save_folder=" ../../data/mturk_samples/sample_test/", bucket_name="scatterplots"
):
    files = os.listdir(save_folder + "data/")

    s3_folder = save_folder.split("/")[-2]

    tuples = []
    s3 = boto3.resource("s3")
    # bucket = s3.Bucket(bucket_name)

    try:
        s3.meta.client.head_bucket(Bucket=bucket_name)
    except botocore.exceptions.ClientError as e:
        print("We could not open the bucket for whatever reason.")
        print(e)

    for i, file in enumerate(files):
        if file.endswith(".csv"):
            try:
                print("Processing file {}".format(file))
                df = pd.read_csv(save_folder + "data/" + file)
                print(df.shape)
                png = file.split("_1-2.csv")[0] + ".png"
                classes = int(len(set(df["class"])))

                for j in range(classes // 4):
                    tuples.append(
                        [
                            "https://scatterplots.s3.eu-central-1.amazonaws.com/"
                            + s3_folder
                            + "/"
                            + png,
                            classes,
                            j,
                        ]
                    )

                if classes // 4 == 0:
                    tuples.append(
                        [
                            "https://scatterplots.s3.eu-central-1.amazonaws.com/"
                            + s3_folder
                            + "/"
                            + png,
                            classes,
                            0,
                        ]
                    )

                # s3.meta.client.upload_file(
                #     Filename = save_folder + 'figures/' + png, Bucket = bucket_name,
                #     Key = s3_folder + '/' + png)

            except Exception as e:
                print("There was an error processing file {}".format(file))
                print(e)

    tuples = np.array(tuples)
    print("Tuple size is {}".format(tuples.shape))

    ddf = pd.DataFrame(
        {
            "image_url": tuples[:, 0],
            "i": tuples[:, 1].astype(int),
            "class_set": tuples[:, 2].astype(int),
        }
    )

    ddf.loc[ddf["class_set"] < 1, :].to_csv(save_folder + "index.csv", index=False)
