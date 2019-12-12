import ray
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding, MDS
import umap
import seaborn as sns
from descartes import PolygonPatch
import matplotlib.pyplot as plt


from SepMe.processing.occfiltering import (
    get_circles,
    remove_circles_by_partition,
    remove_outlier,
)


def underscore_reducer(k1, k2):
    if k1 is None:
        return str(k2)
    else:
        return str(k1) + "_" + str(k2)


def preprocess_df(df, dims, size=0.5, occlusion=0.1, save=None, sort=True):
    ddf = df[dims + ["class"]].copy()
    ddf = ddf.sort_values(["class"])
    # print(ddf.head())

    # print(ddf.shape)
    classes = list(set(ddf["class"]))
    if len(classes) > 10:
        print("Classes: {}".format(len(classes)))
        ddf.columns = ["x", "y", "orig_class"]  # rename cols
        merge_classes = list(
            ddf[["orig_class", "x"]]
            .groupby(["orig_class"])
            .count()
            .sort_values(["x"], ascending=True)
            .head(len(classes) - 9)
            .index
        )

        ddf["class"] = [
            row if row not in merge_classes else -1
            for i, row in ddf["orig_class"].items()
        ]

        le = LabelEncoder()
        ddf["class"] = le.fit_transform(ddf["class"])

    else:
        ddf.columns = ["x", "y", "class"]  # rename cols
        le = LabelEncoder()
        ddf["orig_class"] = ddf["class"]
        ddf["class"] = le.fit_transform(ddf["orig_class"])

    ddf = remove_outlier(ddf, [0.05, 0.95])  # remove outliers
    ddf.iloc[:, :2] = minmax_scale(ddf.iloc[:, :2])  # scale between 0 and 1

    if sort is True:
        ddf = ddf.sort_values(["x", "y"])  # sort

    # print(ddf.shape)

    ddf.reset_index(inplace=True, drop=True)  # reset_index

    # filter occluded circles
    buff = ((max(ddf.x) - min(ddf.x)) / (np.sqrt(len(ddf.x)) * 2)) * size
    if 250 < len(ddf) < 500:
        buff = 1.1 * buff
    elif 500 < len(ddf) < 700:
        buff = 1.2 * buff
    elif len(ddf) > 700:
        buff = 1.5 * buff
    elif len(ddf) < 70:
        buff = 0.8 * buff

    circles = get_circles(ddf, buff)
    rem_indexes = remove_circles_by_partition(ddf, circles, 2 * buff, occlusion)
    ddf = ddf.drop(rem_indexes)
    circles = circles.drop(rem_indexes)

    if save is not None:
        ddf.to_csv(save, index=False)

    return ddf, circles


def plot_colored_circles(ax, df, circles, class_name="class", specific_colors=-1):

    for i, row in df.iterrows():
        if specific_colors == -1:
            ax.add_patch(
                PolygonPatch(
                    circles[i],
                    fc=sns.color_palette("colorblind")[int(row[class_name])],
                    ec="none",
                    alpha=1,
                )
            )
        else:
            if int(row[class_name]) == 0:
                ax.add_patch(
                    PolygonPatch(circles[i], fc="lightgray", ec="none", alpha=1,)
                )
            else:
                ax.add_patch(
                    PolygonPatch(
                        circles[i],
                        fc=sns.color_palette("colorblind")[int(specific_colors)],
                        ec="none",
                        alpha=1,
                    )
                )

    ax.autoscale()
    ax.set_aspect("equal", "datalim")

    return ax


def get_dimred_data(df, input_folder, save_folder, fig_folder):
    found = 0
    not_found = []
    names = ["_".join(row.split("_")[:-1]) for row in list(df["index"])]
    names = list(set(names))

    for i, nn in enumerate(names):

        print("Processing file {}/{}. Name: {}".format(i, len(names), nn))

        method = nn.split("_")[-1]
        # print(method)
        if method in {"PCA", "RobPCA"}:
            name = nn + "_data.csv"
        else:
            name = nn + "_2.csv"

        # print(name)

        try:
            data = pd.read_csv(input_folder + method + "/" + name)
            found += 1

            dim_x = str(df.loc[i, "dim_x"])
            dim_y = str(df.loc[i, "dim_y"])

            file_name = save_folder + nn + "_" + dim_x + "-" + dim_y + ".csv"

            if os.path.exists(file_name) is True:
                print("File was already processed. Skipping: {}".format(file_name))
                continue

            ddf, circles = preprocess_df(
                data, [dim_x, dim_y], size=0.5, occlusion=0.1, save=file_name, sort=True
            )

            fig, ax = plt.subplots(figsize=(20, 20))
            ax.tick_params(axis="both", which="major", labelsize=20)
            plot_colored_circles(ax, ddf, circles)
            fig.savefig(fig_folder + names[i] + ".pdf")
            plt.close("all")

        except FileNotFoundError:
            # print(nn)
            print("File not found: {}".format(name))
            not_found.append(name)

    return


# @ray.remote
def add_dimreds(orig_dir, save_dir, file):
    if os.path.exists(save_dir + file):
        # print('File was already processed. Skipping: {}'.format(save_dir + file))
        return pd.DataFrame({})

    df = pd.read_csv(orig_dir + file)

    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    ndf = df.select_dtypes(include=numerics)
    ndf = ndf.iloc[:, :-1].copy()

    print("Applying UMAP..")
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ndf)
    df["umap1"] = embedding[:, 0]
    df["umap2"] = embedding[:, 1]

    print("Applying PCA..")
    pca = PCA(n_components=3, whiten=True)
    embedding = pca.fit_transform(ndf)
    df["pca1"] = embedding[:, 0]
    df["pca2"] = embedding[:, 1]

    print("Applying Isomap..")
    iso = Isomap(n_components=2, n_neighbors=3)
    embedding = iso.fit_transform(ndf)
    df["iso1"] = embedding[:, 0]
    df["iso2"] = embedding[:, 1]

    print("Applying LLE..")
    lle = LocallyLinearEmbedding(n_components=2, eigen_solver="dense")
    embedding = lle.fit_transform(ndf)
    df["lle1"] = embedding[:, 0]
    df["lle2"] = embedding[:, 1]

    print("Applying tSNE..")
    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(ndf)
    df["tsn1"] = embedding[:, 0]
    df["tsn2"] = embedding[:, 1]

    print("Applying non-metric MDS..")
    mds = MDS(n_components=2, metric=False)
    embedding = mds.fit_transform(ndf)
    df["mds1"] = embedding[:, 0]
    df["mds2"] = embedding[:, 1]

    dr_cols = [
        "pca1",
        "pca2",
        "iso1",
        "iso2",
        "tsn1",
        "tsn2",
        "lle1",
        "lle2",
        "umap1",
        "umap2",
        "mds1",
        "mds2",
    ]

    df = df[dr_cols + list(ndf.columns) + ["class"]]
    df.loc[:, dr_cols + ndf.columns] = minmax_scale(df.loc[:, dr_cols + ndf.columns])

    print(df.columns)
    df.to_csv(save_dir + file, index=False)

    return df


# @ray.remote
def save_and_plot_all_dimensions(file, orig_dir, save_dir, fig_dir, class_cols):
    df = pd.read_csv(orig_dir + file)

    c1s = []
    for i, c1 in enumerate(df.columns):
        if not np.issubdtype(df[c1].dtype, np.number) or (c1 in class_cols) or i > 25:
            continue

        c1s.append(c1)
        for i, c2 in enumerate(df.columns[:-1]):
            if (
                (not np.issubdtype(df[c1].dtype, np.number))
                or (c2 in c1s)
                or (c2 in class_cols)
                or i > 25
            ):
                continue

            file_name = (
                save_dir + file.split(".csv")[0] + "_{}-{}".format(c1, c2) + ".csv"
            )
            fig_name = (
                fig_dir + file.split(".csv")[0] + "_{}-{}".format(c1, c2) + ".png"
            )

            if os.path.exists(file_name) is True:
                # print('File was already processed. Skipping: {}'.format(file_name.split('/')[-1]))
                continue

            try:
                print("{} (len: {}): ({}-{})".format(file, len(df), c1, c2))
                ddf, circles = preprocess_df(
                    df, [c1, c2], size=0.5, occlusion=0.1, save=file_name, sort=True
                )
                fig, ax = plt.subplots(figsize=(20, 20))

                plot_colored_circles(ax, ddf, circles)
                ax.tick_params(axis="both", which="major", labelsize=30)

                fig.savefig(fig_name)
                plt.close("all")

            except Exception as e:
                print(
                    "File {} was not processed for columns ({},{}).".format(
                        file, c1, c2
                    )
                )
                print(e)
                print("")


# @ray.remote
def process_one_dimred(i, names, nn, input_folder, df, save_folder, fig_folder):

    method = nn.split("_")[-1]
    # print(method)
    if method in {"PCA", "RobPCA"}:
        name = nn + "_data.csv"

    elif method in ["tSNE"]:
        name = nn + "_2.csv"
        name = name.replace("tSNE", "TSNE")
    else:
        name = nn + "_2.csv"

    try:
        data = pd.read_csv(input_folder + method + "/" + name)

        dim_x = str(df.loc[i, "dim_x"])
        dim_y = str(df.loc[i, "dim_y"])

        file_name = save_folder + nn + "_" + dim_x + "-" + dim_y + ".csv"

        if os.path.exists(fig_folder + names[i] + ".png") is True:
            print("---")
            # print("File was already processed. Skipping: {}".format(file_name))
            # print("Processed file {}/{}. Name: {}".format(i, len(names), nn))
            return

        ddf, circles = preprocess_df(
            data, [dim_x, dim_y], size=0.5, occlusion=0.1, save=file_name, sort=True
        )

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.tick_params(axis="both", which="major", labelsize=20)
        plot_colored_circles(ax, ddf, circles)
        fig.savefig(fig_folder + names[i] + ".png")
        plt.close("all")

        print("Processed file {}/{}. Name: {}".format(i, len(names), nn))

    except FileNotFoundError:
        # print(nn)
        print("File not found: {}".format(name))
