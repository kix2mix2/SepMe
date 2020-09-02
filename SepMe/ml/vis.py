import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.spatial


def grid_scat(
    data,  # data frame containing x,y,hue columns as well as
    # a path column which leads to where the images for each coordinate are saved.
    x,  # name of the x coordinate column to plot
    y,  # name of the y coordinate column to plot
    hue="human_rating.mean",  # color of the border of the
    name="scatter_PCA_grid.pdf",  # the name of the figure you save
    figsize=(80, 80),  # size of the figure
    gridsize=(40, 40),  # size of the grid (40x40 cells in total by default)
    n=0,  # this is for overplotting. several scatterplots may be plotted on top of each,
    # this decides which ones to plot.
):
    x = data[x]
    y = data[y]

    size_x = np.int(gridsize[0])
    size_y = np.int(gridsize[1])

    # get grid
    cen = get_grid(x, y, size_x=size_x, size_y=size_y)

    # calculate which grid cell is closest to each dataploint
    results2 = do_kdtree(cen, np.dstack([x, y]))

    # save new location in your dataframe
    data["grid_loc"] = results2[0]
    data["gx"] = cen[results2[0]][:, 0]
    data["gy"] = cen[results2[0]][:, 1]

    # select datapoints to plot
    df_placed = data.groupby("grid_loc").nth(n).reset_index()

    # print(df_placed.loc[df_placed['y']==1,'filename'])
    # mini_df = df_placed.loc[df_placed['y']==1,:]
    # ff,ax1 = plt.subplots(figsize=(10,10))
    # sns.barplot(x = "filename", y = "spread", data = mini_df, capsize = .2, ax=ax1)

    df_placed = df_placed.sort_values(["gx", "gy"], ascending=False)

    # plot the scatterplot of images
    scat_of_scat(
        list(df_placed["gx"]),
        list(df_placed["gy"]),
        df_placed["path"],
        colors=np.round(df_placed[hue] * 100),
        figsize=figsize,
        name=name,
        steps=100,
    )

    return data


def do_kdtree(combined_x_y_arrays, points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes


def get_grid(x, y, size_x=40, size_y=40):
    min_x = np.min(x)
    max_x = np.max(x)
    min_y = np.min(y)
    max_y = np.max(y)

    grid_x = np.linspace(min_x, max_x, size_x)
    cen_x = np.linspace(min_x, max_x, size_x * 2)[
        [i for i in range(size_x * 2) if i % 2 == 1]
    ]
    grid_y = np.linspace(min_y, max_y, size_y)
    cen_y = np.linspace(min_y, max_y, size_y * 2)[
        [i for i in range(size_y * 2) if i % 2 == 1]
    ]

    grid = np.array([grid_x, grid_y]).T

    xx, yy = np.meshgrid(cen_x, cen_y)
    # xx, yy

    yy = np.reshape(yy, (1, np.product(yy.shape)))
    xx = np.reshape(xx, (1, np.product(xx.shape)))

    # fig, ax = plt.subplots(figsize=(5,5))
    # ax.scatter(xx, yy)

    cen = np.dstack([xx, yy])[0]
    return cen


def getImage(path):
    return OffsetImage(plt.imread(path))


def scat_of_scat(
    x, y, paths, colors, name="scatter_PCA.pdf", figsize=(50, 50), steps=100
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y)

    plt.rcParams.update({"font.size": 12})

    mmin = np.min(colors)
    mmax = np.max(colors)
    print(mmin)
    print(mmax)

    i = 0
    for x0, y0, path, c in zip(x, y, paths, colors):
        if i % 100 == 0:
            print(i)
        if os.path.exists(path):
            ab = AnnotationBbox(getImage(path), (x0, y0), frameon=True)
            ab.patch.set_linewidth(4)
            ab.patch.set_edgecolor(plt.cm.RdBu((np.clip(c, mmin, mmax) - 1) / steps))
            ab.patch.set_facecolor(plt.cm.RdBu((np.clip(c, mmin, mmax) - 1) / steps))
            ax.add_artist(ab)

        i += 1

    fig.savefig(name)


def resize_and_save(
    file,
    save_dir="../../data/mturk_samples/task/baby_figs/",
    scale_percent=5,
    size=None,
):
    # USAGE
    # path = '../../data/mturk_samples/task/figures/'
    # paths = [path + row['filename'] + '.png'  for i,row in df_agg_abs.iterrows()]

    # for p in paths:
    #     resize_and_save(p)

    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # percent of original size
    if size is None:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        print((width, height))

    else:
        dim = size

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    baby_path = save_dir + file.split("/")[-1]
    cv2.imwrite(baby_path, resized)


def plot_violins(df, ax, x="category", y="human_rating", hue="type"):
    df = df.sort_values([x])

    sns.violinplot(
        x=x,
        y=y,
        hue=hue,
        split=True,
        inner="quart",
        palette={"semantic": "lightblue", "abstract": "lightyellow"},
        data=df,
        ax=ax[0],
    )

    sns.boxplot(
        x=x, y=y, hue=hue, palette=["lightblue", "lightyellow"], data=df, ax=ax[1]
    )


def plot_corr_matrix(df, figsize=(11, 9)):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
