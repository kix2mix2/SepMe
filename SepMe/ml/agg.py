import pandas as pd
import seaborn as sns
from scipy.stats import sem, t
import numpy as np
from sklearn.preprocessing import minmax_scale


def add_interval(
    df,
    intervals=[0, 0.2, 0.6, 0.8, 1.00],
    col_name="interval_4",
    target_interval="human_rating.mean",
):
    df[col_name] = "[{},{}]".format(intervals[0], intervals[1])
    for i, ints in enumerate(intervals):
        if i != len(intervals) - 1:
            df.loc[
                df[target_interval].between(intervals[i], intervals[i + 1]), col_name
            ] = "[{},{}]".format(
                np.round(intervals[i], 2), np.round(intervals[i + 1], 2)
            )

    return df


def aggregate(df, by=["filename", "type", "class"]):
    df_agg = (
        df.groupby(by).agg(
            {
                "human_rating": ["mean", "count", sem],
                "expert_rating": ["first", "last", "mean"],
            }
        )
    ).sort_values(by)

    try:
        df_agg.columns = [
            "%s%s" % (a, ".%s" % b if b else "") for a, b in df_agg.columns
        ]
        col = "human_rating.sem"
    except Exception:
        print("meh")

    df_agg1 = (df.groupby(by).mean()).sort_values(by)

    df_agg = pd.concat([df_agg, df_agg1], axis=1)
    df_agg = df_agg.sort_values(by)
    df_agg = df_agg.reset_index()

    confidence = 0.95
    h = df_agg["human_rating.sem"] * t.ppf(
        (1 + confidence) / 2, df_agg["human_rating.count"] - 1
    )

    df_agg["start"] = df_agg["human_rating.mean"] - h
    df_agg["end"] = df_agg["human_rating.mean"] + h
    df_agg["spread"] = df_agg["end"] - df_agg["start"]

    df_agg = df_agg.dropna(axis=1)
    df_agg = df_agg.sort_values(["expert_rating.first"] + by)

    df_agg["half_interval"] = df_agg["spread"] / 2

    return df_agg


def reshape_sepme(sepme_path, save=None):
    sep_df = pd.read_csv(sepme_path)
    sep_df.columns = ["filename"] + list(sep_df.columns)[1:]
    print(sep_df.shape)
    sep_df.index = sep_df.filename
    sep_df = sep_df.drop(["filename"], axis=1).stack().reset_index()
    sep_df.columns = ["filename", "method", "value"]

    stuff = [row.split("_") for i, row in sep_df["method"].items()]

    new_stuff = []
    for s in stuff:
        ns = []

        if len(s) <= 2:
            ns = ["del", "n/a", "mcec", -1]

        else:
            ns.append(s[0])  # graph_type
            try:
                ns.append(float(s[1]))  # graph_param
                ns.append(s[2])  # graph_purity
                if len(s) == 4:
                    try:
                        ns.append(int(s[3]))  # class
                    except ValueError:
                        ns.append(-1)  # class

                else:
                    ns.append(-1)  # class

                if len(s) == 5:
                    ns[2] += "_" + s[4]  # purity

            except ValueError:
                # print(s[1])
                ns.append("n/a")
                ns.append(s[1])  # graph_purity
                try:
                    ns.append(int(s[2]))  # class
                except ValueError:
                    ns.append(-1)  # class
                if len(s) == 4:
                    ns[2] += "_" + s[3]

        new_stuff.append(ns)
        # print('s {} - ns {}'.format(s,ns))
        if len(ns) != 4:
            print(ns)

    sep_df["graph_type"] = [s[0] for s in new_stuff]
    sep_df["graph_param"] = [s[1] for s in new_stuff]
    sep_df["graph_purity"] = [s[2] for s in new_stuff]
    sep_df["class"] = [s[3] for s in new_stuff]

    sep_df.groupby(["graph_purity"]).count()

    sep_df = sep_df.loc[sep_df["class"].between(0, 8)]

    sep_df = sep_df.drop(["method"], axis=1)

    sep_df["graph"] = [
        "{}_{}_{}".format(row["graph_type"], row["graph_param"], row["graph_purity"])
        for i, row in sep_df.iterrows()
    ]
    sep_df["idx"] = [
        "{}:{}".format(row["filename"], row["class"]) for i, row in sep_df.iterrows()
    ]

    if save is not None:
        try:
            sep_df.to_csv(save, index=False)
        except Exception:
            print(
                "Save attribute should be the name of a csv file. You may append a path to save it elsewhree."
            )
    return sep_df


def pivot_sepme(sep_df):
    df = sep_df.pivot(index="idx", columns="graph", values="value").reset_index()
    df["class"] = [int(row["idx"].split(":")[-1]) + 1 for i, row in df.iterrows()]
    df["idx"] = [
        row["idx"].split("1-2")[0] + str(row["class"]) for i, row in df.iterrows()
    ]
    df = df.dropna(thresh=0.8 * len(df), axis=1).dropna(
        thresh=0.5 * df.shape[1], axis=0
    )
    return df


def reshape_humandf(res_path, index=["fileName", "type", "1v1", "phase"], drop=True):
    res_df = pd.read_csv(res_path)

    cols = [
        "fileName",
        "WorkerId",
        "SubmitTime",
        "total_recordings",
        "type",
        "1v1",
        "phase",
        "Reward",
        "sep1",
        "sep2",
        "sep3",
        "sep4",
        "sep5",
        "sep6",
        "sep7",
        "sep8",
        "pass1",
        "pass2",
        "pass3",
        "pass4",
    ]
    seps = ["sep1", "sep2", "sep3", "sep4"]
    passes = ["pass1", "pass2", "pass3", "pass4"]
    res_df = res_df[cols]

    res_df.columns = ["filename"] + list(res_df.columns)[1:]

    for i in range(1, 5):
        res_df.loc[res_df["sep{}".format(i)].isna(), "pass{}".format(i)] = np.nan

    res_df = res_df.set_index(index)
    df1 = res_df[seps].stack().reset_index()
    df2 = res_df[passes].stack().reset_index()

    df1.columns = index + ["class", "human_rating"]
    df2.columns = index + ["class", "pass"]

    df1["class"] = [row.split("sep")[-1] for i, row in df1["class"].items()]
    df2["class"] = [row.split("pass")[-1] for i, row in df2["class"].items()]

    df1["idx"] = df1["filename"] + "_" + df1["class"].astype(str)
    df2["idx"] = df2["filename"] + "_" + df2["class"].astype(str)

    df2 = df2.drop(index + ["class"], axis=1)

    df = pd.concat([df1, df2[["pass"]]], axis=1)

    df["human_rating"] = minmax_scale(df["human_rating"])

    ## comment out in the future
    if drop:
        df = df.loc[df["1v1"] is False]
        df = df.loc[df["pass"] == 1]
        df = df.drop(["pass", "1v1"], axis=1)

    return df


def append_expertdf(df, expert_path="../../data/orig_data/human_reduced_results.csv"):
    expert_df = pd.read_csv(expert_path)
    expert_df = expert_df[["index", "M", "A"]]
    expert_df = expert_df.set_index("index").stack().reset_index()
    expert_df.columns = ["index", "expert_name", "expert_rating"]
    df = df.merge(expert_df, left_on="idx", right_on="index")
    df = df.drop(["index", "class"], axis=1)
    return df
