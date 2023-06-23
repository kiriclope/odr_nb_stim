import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.io import loadmat

from decoder import get_pipeline
from my_bootstrap import my_boots_ci

import numpy.ma as ma

from analysis import *

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7.5
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


def norm_std(X, axis):
    return np.nanstd(X, axis) / np.nanmean(X, axis)


if __name__ == "__main__":

    data = load_file()
    df = get_df(data)

    # df = df.dropna()
    df = df[df.S1 != -1]
    df = df[df.S2 != -1]
    df = df[df.S2 != 2]

    df = df[df.task == 0]

    epochs = ["cue_rate", "delay1_rate", "sample_rate", "delay2_rate"]
    epoch = "delay2_rate"
    stim = "S1"

    df_off = df[(df.NB == 0) & (df.S1 == 1)]
    X_off, y_off = get_X_y(df_off, epoch, stim)
    X_off_p = np.nanstd(X_off, axis=0)
    X_off_p = np.nanstd(X_off, axis=0) / np.nanmean(X_off, axis=0)

    ci_off_p = my_boots_ci(
        X_off_p, norm_std, n_samples=10000, method="BCa", alpha=0.05, vectorized=True
    )

    df_on = df[(df.NB == 1) & (df.S1 == 1)]
    X_on, y_on = get_X_y(df_on, epoch, stim)
    # X_on_p = np.nanstd(X_on, axis=0)
    X_on_p = np.nanstd(X_on, axis=0) / np.nanmean(X_on, axis=0)

    ci_on_p = my_boots_ci(
        X_on_p, norm_std, n_samples=10000, method="BCa", alpha=0.05, vectorized=True
    )

    df_off = df[(df.NB == 0) & (df.S1 == 5)]
    X_off, y_off = get_X_y(df_off, epoch, stim)
    # X_off_np = np.nanstd(X_off, axis=0)
    X_off_np = np.nanstd(X_off, axis=0, ddof=1) / np.nanmean(X_off, axis=0)

    ci_off_np = my_boots_ci(
        X_off_np, norm_std, n_samples=10000, method="BCa", alpha=0.05, vectorized=True
    )

    print(np.array(ci_off_np).shape)

    df_on = df[(df.NB == 1) & (df.S1 == 5)]
    X_on, y_on = get_X_y(df_on, epoch, stim)
    # X_on_np = np.nanstd(X_on, axis=0)
    X_on_np = np.nanstd(X_on, axis=0, ddof=1) / np.nanmean(X_on, axis=0)

    ci_on_np = my_boots_ci(
        X_on_np, norm_std, n_samples=10000, method="BCa", alpha=0.05, vectorized=True
    )

    plt.bar([0, 1], [np.nanmean(X_off_np), np.nanmean(X_on_np)], width=1, color=pal)
    plt.bar([2, 3], [np.nanmean(X_off_p), np.nanmean(X_on_p)], width=1, color=pal)

    plt.errorbar(
        [0], np.nanmean(X_off_np), yerr=np.array(ci_off_np)[:, np.newaxis], color="k"
    )

    plt.errorbar(
        [1], np.nanmean(X_on_np), yerr=np.array(ci_on_np)[:, np.newaxis], color="k"
    )

    plt.errorbar(
        [2], np.nanmean(X_off_p), yerr=np.array(ci_off_p)[:, np.newaxis], color="k"
    )

    plt.errorbar(
        [3], np.nanmean(X_on_p), yerr=np.array(ci_on_p)[:, np.newaxis], color="k"
    )

    plt.xticks([0, 1, 2, 3], ["OFF", "ON", "OFF", "ON"])
    plt.ylim([0.8, 1.5])
    plt.ylabel("Norm std")
    # plt.scatter(X_off, X_on)
    # plt.xlabel("NP")
    # plt.ylabel("P")

    # plt.xlabel("OFF")
    # plt.ylabel("ON")

    # corr = ma.corrcoef(ma.masked_invalid(X_off), ma.masked_invalid(X_on))[0, 1]
    # print("Correlation coefficient:", corr)

    # # Add the correlation coefficient as a text annotation
    # plt.annotate(f"Correlation = {corr:.2f}", xy=(0.05, 0.9), xycoords="axes fraction")

    # plt.hist(X_off, bins="auto", histtype="step", color=pal[0], density=True)
    # plt.hist(X_on, bins="auto", histtype="step", color=pal[1], density=True)

    # plt.vlines(np.nanmean(X_off), 0, 1, color=pal[0], ls="--")
    # plt.vlines(np.nanmean(X_on), 0, 1, color=pal[1], ls="--")

    print("np", np.nanmean(X_off_np), np.nanmean(X_on_np))
    print("p", np.nanmean(X_off_p), np.nanmean(X_on_p))
