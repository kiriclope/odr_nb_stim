import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import t

import statsmodels.api as sm
from my_bootstrap import my_boots_ci

from analysis import *

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7.5
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


def ci_student(data):
    # Calculate the sample mean and standard deviation
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)

    # Set the desired confidence level (e.g., 95%)
    confidence_level = 0.95

    # Calculate the degrees of freedom
    degrees_of_freedom = len(data) - 1

    # Calculate the critical value
    critical_value = t.ppf((1 + confidence_level) / 2, df=degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = critical_value * sample_std / np.sqrt(len(data))

    # Calculate the confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    ci = np.vstack((lower_bound, upper_bound))
    print("ci", ci.shape, ci)
    return ci


def norm_var(X, axis=0):
    # X_scale = z_score(X, 1)
    # print(X.shape, X_scale.shape)
    # return np.nanstd(X, axis, ddof=0) / np.nanmean(X, axis)
    # return np.nanstd(X, axis, ddof=0) / np.nanmean(X, axis)
    return np.nanmean(X, axis)


if __name__ == "__main__":

    data = load_file()
    df = get_df(data)

    # df = df.dropna()
    # df = df[df.S1 != -1]
    df = df[df.S2 != -1]
    # df = df[df.S2 != 2]

    # df = df[df.task == 0]
    # df["delay2_rate"] = z_score(df["delay2_rate"], axis=0)
    # df["delay1_rate"] = z_score(df["delay1_rate"], axis=0)
    # df["delays_rate"] = z_score(df["delays_rate"], axis=0)

    epochs = ["cue_rate", "delay1_rate", "sample_rate", "delay2_rate"]
    epoch = "delays_rate"

    stim = "S1"

    idx_p = (df.S1 == 1) & (df.task == 0)
    idx_np = (df.S1 == 5) & (df.task == 0)

    # idx_p = ((df.S1 == 1) & (df.task == 0)) | ((df.S2 == 1) & (df.task == 1))
    # idx_np = ((df.S1 == 5) & (df.task == 0)) | ((df.S2 == 5) & (df.task == 1))

    # idx_p = (df.S2 == 1) & (df.task == 1)
    # idx_np = (df.S2 == 5) & (df.task == 1)

    df_off = df[(df.NB == 0) & idx_p]
    X_off, y_off = get_X_y(df_off, epoch, stim)
    X_off_p = norm_var(X_off)

    ci_off_p = my_boots_ci(
        X_off_p,
        np.nanmean,
        n_samples=10000,
        method="BCa",
        alpha=0.05,
        # vectorized=True,
        means=np.nanmean(X_off_p),
    )

    # ci_off_p = ci_student(X_off_p)

    df_on = df[(df.NB == 1) & idx_p]
    X_on, y_on = get_X_y(df_on, epoch, stim)
    X_on_p = norm_var(X_on)

    ci_on_p = my_boots_ci(
        X_on_p,
        np.nanmean,
        n_samples=10000,
        method="BCa",
        alpha=0.05,
        # vectorized=True,
        means=np.nanmean(X_on_p),
    )

    # ci_on_p = ci_student(X_on_p)

    df_off = df[(df.NB == 0) & idx_np]

    X_off, y_off = get_X_y(df_off, epoch, stim)
    X_off_np = norm_var(X_off)

    ci_off_np = my_boots_ci(
        X_off_np,
        np.nanmean,
        n_samples=10000,
        method="BCa",
        alpha=0.05,
        # vectorized=True,
        means=np.nanmean(X_off_np),
    )

    # ci_off_np = ci_student(X_off_np)

    print(np.array(ci_off_np).shape)

    df_on = df[(df.NB == 1) & idx_np]
    X_on, y_on = get_X_y(df_on, epoch, stim)
    X_on_np = norm_var(X_on)

    ci_on_np = my_boots_ci(
        X_on_np,
        np.nanmean,
        n_samples=10000,
        method="BCa",
        alpha=0.05,
        # vectorized=True,
        means=np.nanmean(X_on_np),
    )

    # ci_on_np = ci_student(X_on_np)

    plt.figure("sd")
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
    plt.ylabel("Norm. Standard Deviation")

    print("np", np.nanmean(X_off_np), np.nanmean(X_on_np))
    print("p", np.nanmean(X_off_p), np.nanmean(X_on_p))

    _, pval = ttest_rel(X_off_np, X_on_np, axis=0, nan_policy="omit")
    print("pval NP", pval)

    _, pval = ttest_rel(X_off_p, X_on_p, axis=0, nan_policy="omit")
    print("pval P", pval)

    df = pd.DataFrame()
    df["neuron"] = np.hstack(
        (np.arange(233), np.arange(233), np.arange(233), np.arange(233))
    )
    df["norm_std"] = np.hstack((X_off_np, X_on_np, X_off_p, X_on_p))
    df["P"] = np.hstack((np.zeros(466), np.ones(466))).astype(int)
    df["NB"] = np.hstack(
        (np.zeros(233), np.ones(233), np.zeros(233), np.ones(233))
    ).astype(int)

    df = df.dropna()
    model = sm.MixedLM.from_formula(
        "norm_std ~ P * NB + (0 + P | neuron)", data=df, groups=df["neuron"]
    )
    result = model.fit()

    print(result.summary())
