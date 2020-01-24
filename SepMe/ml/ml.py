import scikitplot as skplt
import matplotlib.pyplot as plt
import random
from scipy.stats import sem, t
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split


def bootstrap_aoc(
    df, bs=50, sample=1000, target_col="human_bin", thresh=0.95, confidence=0.95
):

    aucs = {}
    for col in df.columns[7:-2]:

        if df[col].dtype == "float64":
            # print(col)
            df.loc[:, "a"] = 1 - df[col]
            a = np.array(df.loc[:, target_col].copy())
            b = np.array(df.loc[:, "a"].copy())

            stuff = []
            for i in range(bs):
                idx = random.sample(list(np.arange(len(a))), sample)
                stuff.append(roc_auc_score(a[idx], b[idx]))

            n = len(stuff)
            m = np.mean(stuff)
            std_err = sem(stuff)
            h = std_err * t.ppf((1 + confidence) / 2, n - 1)

            aucs[col] = {}
            aucs[col]["mean"] = m
            aucs[col]["start"] = m - h
            aucs[col]["end"] = m + h

            if aucs[col]["mean"] > thresh:
                print(col)
                b = np.array(df.loc[:, [col, "a"]].copy())
                skplt.metrics.plot_roc(a, b)
                plt.show()
