import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.preprocessing import minmax_scale


def aggregate_batch(df):
    df_agg = (
        df.groupby(["HITname"])
        .agg(
            {
                "sep1": {"mean", "std", "count", scipy.stats.sem},
                "sep2": {"mean", "std", "count", scipy.stats.sem},
                "sep3": {"mean", "std", "count", scipy.stats.sem},
                "sep4": {"mean", "std", "count", scipy.stats.sem},
                "WorkTimeInSeconds": {"min", "max", "median", "mean"},
                "mp.1": "min",
                "ap.1": "min",
                "mp.2": "min",
                "ap.2": "min",
                "mp.3": "min",
                "ap.3": "min",
                "mp.4": "min",
                "ap.4": "min",
            }
        )
        .reset_index()
    )

    df_agg.columns = ["%s%s" % (a, ".%s" % b if b else "") for a, b in df_agg.columns]
    df_agg.columns = [
        a.replace("Answer.Separability", "sep")
        if a.startswith("Answer.Separability")
        else a
        for a in list(df_agg.columns)
    ]

    confidence = 0.95
    for i in range(1, 5):
        df_agg["sep{}.h".format(i)] = df_agg["sep{}.sem".format(i)] * scipy.stats.t.ppf(
            (1 + confidence) / 2.0, df_agg["sep{}.count".format(i)] - 1
        )
        df_agg.drop(
            ["sep" + str(i) + ".std", "sep{}.sem".format(i)], axis=1, inplace=True
        )

    df_agg = df_agg.sort_values(["HITname"])

    return df_agg


def append_test_data(batch_file, check_file):
    df = pd.read_csv(batch_file)
    df["HITname"] = [
        row.split("/")[-1].split(".png")[0] for i, row in df["Input.image_url"].items()
    ]
    df.columns = [
        a.replace("Answer.Separability", "sep")
        if a.startswith("Answer.Separability")
        else a
        for a in list(df.columns)
    ]
    dfh = pd.read_csv(check_file)
    dfh = dfh.loc[dfh["class"] <= 4, :]
    dfh.head()
    dfh["mp"] = minmax_scale(dfh["M"], feature_range=(0, 100))
    dfh["ap"] = minmax_scale(dfh["A"], feature_range=(0, 100))

    dfh1 = dfh.pivot(
        index="fileName", columns="class", values=["mp", "ap"]
    ).reset_index()
    dfh1.columns = ["%s%s" % (a, ".%s" % b if b else "") for a, b in dfh1.columns]

    df = dfh1.merge(df, left_on="fileName", right_on="HITname")

    drop = [
        "HITTypeId",
        "Title",
        "Description",
        "Keywords",
        "Reward",
        "CreationTime",
        "MaxAssignments",
        "RequesterAnnotation",
        "AssignmentDurationInSeconds",
        "AutoApprovalDelayInSeconds",
        "Expiration",
        "NumberOfSimilarHITs",
        "LifetimeInSeconds",
        "AssignmentId",
        "AssignmentStatus",
        "AcceptTime",
        "SubmitTime",
        "AutoApprovalTime",
        "ApprovalTime",
        "RejectionTime",
        "RequesterFeedback",
        "LifetimeApprovalRate",
        "HITId",
    ]
    df.drop(drop, axis=1, inplace=True)

    return df


def append_cristina(
    batch_file, check_file="../data/mturk_samples/training_abstract/index.csv"
):

    df = pd.read_csv(batch_file)
    df["HITname"] = [
        row.split("/")[-1].split(".png")[0] for i, row in df["Input.image_url"].items()
    ]
    df = df.sort_values(["HITname"])
    df.columns = [
        a.replace("Answer.Separability", "sep")
        if a.startswith("Answer.Separability")
        else a
        for a in list(df.columns)
    ]

    idx = pd.read_csv(check_file)
    idx["HITname"] = [a.split("/")[-1].split(".png")[0] for a in idx.image_url]
    idx["mp.1"] = [0, 60, 60, 60, 0, 60, 0, 60]
    idx["ap.1"] = [40, 100, 100, 100, 100, 100, 40, 100]

    idx["mp.2"] = [0, 60, 60, 60, 0, 60, 0, 60]
    idx["ap.2"] = [40, 100, 100, 100, 50, 100, 40, 100]

    idx["mp.3"] = [0, 60, 60, 40, 0, 60, 0, 60]
    idx["ap.3"] = [40, 100, 100, 90, 50, 100, 40, 100]

    idx["mp.4"] = [0, None, None, None, 0, 60, 0, None]
    idx["ap.4"] = [40, None, None, None, 50, 100, 40, None]
    df = df.merge(idx, on="HITname")

    return df


def get_plots(df, df_agg, figsize=(10, 20), alpha=0.6, col="white", con_col="#cc78bc"):
    fig, axes = plt.subplots(4, 1, figsize=figsize)

    for j in range(4):
        # print('{}: {},{}'.format(j,j//2,j%2))

        sns.boxplot(
            y="HITname",
            x="sep{}".format(j + 1),
            data=df,
            orient="h",
            ax=axes[j],
            color="#D9D9D9",
            boxprops=dict(alpha=alpha),
        )

        for i, artist in enumerate(axes[j].artists):
            # Set the linecolor on the artist to the facecolor, and set the facecolor to None

            artist.set_edgecolor(col)
            for jj in range(i * 6, i * 6 + 6):
                line = axes[j].lines[jj]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        for i in range(len(df_agg)):
            # print(df_agg.loc[i, 'HITname'])
            axes[j].plot(
                [
                    df_agg.loc[i, "mp.{}.min".format(j + 1)],
                    df_agg.loc[i, "ap.{}.min".format(j + 1)],
                ],
                [i, i],
                "o-",
                markersize=15,
                color="#ece133",
                linewidth=15,
                alpha=0.1,
            )

            axes[j].plot(
                [
                    df_agg.loc[i, "sep{}.mean".format(j + 1)]
                    - df_agg.loc[i, "sep{}.h".format(j + 1)],
                    df_agg.loc[i, "sep{}.mean".format(j + 1)]
                    + df_agg.loc[i, "sep{}.h".format(j + 1)],
                ],
                [i, i],
                "-",
                color=con_col,
                linewidth=15,
                alpha=0.6,
            )
            axes[j].plot(
                df_agg.loc[i, "sep{}.mean".format(j + 1)],
                i,
                "|",
                color=con_col,
                markersize=25,
            )

    return axes


def plot_extra(axes, df, df_agg, color="#cc78bc"):
    for j in range(4):
        # print('{}: {},{}'.format(j,j//2,j%2))

        for i in range(len(df_agg)):
            # axes[j].plot([df_agg.loc[i, 'cp{}0.min'.format(j + 1)], df_agg.loc[i, 'cp{}1.min'.format(j + 1)]], [i, i], 'o-',
            #              markersize = 15, color = '#ece133', linewidth = 15, alpha = 0.5)

            axes[j].plot(
                [
                    df_agg.loc[i, "sep{}.mean".format(j + 1)]
                    - df_agg.loc[i, "sep{}.h".format(j + 1)],
                    df_agg.loc[i, "sep{}.mean".format(j + 1)]
                    + df_agg.loc[i, "sep{}.h".format(j + 1)],
                ],
                [i, i],
                "-",
                color=color,
                linewidth=15,
                alpha=0.6,
            )
            axes[j].plot(
                df_agg.loc[i, "sep{}.mean".format(j + 1)],
                i,
                "|",
                color=color,
                markersize=25,
            )


def get_counts(df, p_col, i, total):
    df_agg1 = df.groupby(["HITname"]).agg({"sep{}".format(i): {"count"}}).reset_index()
    df_agg1.columns = ["%s%s" % (a, ".%s" % b if b else "") for a, b in df_agg1.columns]
    df_agg1[p_col] = df_agg1["sep{}.count".format(i)] / total
    df_agg1.drop(["sep{}.count".format(i)], axis=1, inplace=True)
    return df_agg1


def self_clean(df, majority=0.6, total=20, neg=40, pos=60):
    orig = len(df)
    for i in range(1, 5):
        col = "sep{}".format(i)
        ppos = "ppos{}".format(i)
        pneg = "pneg{}".format(i)

        # print(df.shape)

        df_neg = df.loc[df[col] < neg]  # get all rows where score{} is <negative thresh
        df_pos = df.loc[df[col] > pos]  # get all rows where score{} is >positive thresh

        if len(df_neg) > 0:
            df_agg_neg = get_counts(
                df_neg, pneg, i, total
            )  # calculate proportion of negative
            # print(df_agg_neg)
            # print('----')
            df = df.merge(df_agg_neg, on="HITname", how="left")

        else:
            df[pneg] = 0

        if len(df_pos) > 0:

            df_agg_pos = get_counts(
                df_pos, ppos, i, total
            )  # calculate proportion of positive
            # print(df_agg_pos)
            # print('----')

            df = df.merge(df_agg_pos, on="HITname", how="left")

        else:
            df[ppos] = 0

        df[[ppos, pneg]] = df[[ppos, pneg]].fillna(
            0
        )  # if there is a missing value, it means the proportion is 0

        # print('df_agg {}. df_agg_neg {}. df_agg_pos {}'.format(len(df_agg), len(df_agg_neg), len(df_agg_pos)))

        df["pass{}".format(i)] = 1
        df.loc[
            ((df[col] > pos) & (df[pneg] > majority))
            | ((df[col] < neg) & (df[ppos] > majority)),
            "pass{}".format(i),
        ] = 0  # if row is positive but the majority is negative, discar

        # print(df.loc[:,['HITname', col, ppos, pneg,'pass{}'.format(i)] ])
        # print('=======')

    df["passes"] = 0
    for i in range(1, 5):
        df["passes"] += df["pass{}".format(i)]

    print(
        "Retain rate for minimum {} passes is: {}% \n".format(
            2, (len(df.loc[df["passes"] >= 2]) / orig) * 100
        )
    )
    return df


def filter_by_sanity(df):
    for i in range(1, 5):
        df = df.loc[
            df["sep{}".format(i)].between(df["mp.{}".format(i)], df["ap.{}".format(i)])
            | df["sep{}".format(i)].isna()
        ]
        print("Size after filtering by {} is {}".format(i, df.shape))

    return df


def workingtime_stats(df):
    print("Total Workers who took the experiment: {}".format(len(set(df.WorkerId))))
    print("-------")
    print(
        "Average time spent in this experiment: {}".format(
            np.mean(df.WorkTimeInSeconds)
        )
    )
    print(
        "Median time spent in this experiment: {}".format(
            np.median(df.WorkTimeInSeconds)
        )
    )
    print("-------")


def get_worktime_dist(df, bins=8, title=None):
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(111)
    sns.distplot(df.WorkTimeInSeconds, ax=ax, bins=bins, axlabel=title)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(15))

    ll = [0, 15, 30, 60, 120, 180, 240, 300]
    for i, number in enumerate(ll):
        if i == 0:
            continue
        print(
            "Percentage of people between ({},{}): {}".format(
                ll[i - 1],
                ll[i],
                round(
                    len(df.loc[df.WorkTimeInSeconds.between(ll[i - 1], ll[i])])
                    / len(df),
                    2,
                ),
            )
        )


def get_slice_stats(df, mmin=0, mmax=1000, bins=8):
    print("Stats for workers with HITs between {} and {}:".format(mmin, mmax))
    workers = (
        df.groupby(["WorkerId"])
        .agg({"sep1": "count"})
        .reset_index()
        .sort_values(["sep1"], ascending=False)
    )
    worker_ids = list(workers.loc[workers["sep1"].between(mmin, mmax), "WorkerId"])

    workingtime_stats(df.loc[df.WorkerId.isin(worker_ids), :])

    get_worktime_dist(
        df.loc[df["WorkerId"].isin(worker_ids)],
        bins=bins,
        title="workers between {} and {}:".format(mmin, mmax),
    )

    print("===================")
    print("\n\n")
