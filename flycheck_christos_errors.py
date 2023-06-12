import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

from utils import *
from bootstrap import my_boots


# def f_test(group1, group2):
#     f = np.nanvar(group1, ddof=1) / np.nanvar(group2, ddof=1)
#     nun = x.size - 1
#     dun = y.size - 1
#     p_value = 1 - scipy.stats.f.cdf(f, nun, dun)
#     return f, p_value


if __name__ == "__main__":

    THRESH = 30

    IF_CORRECT = True
    CUT_OFF = [np.nan, 0, 45, 90, 180]

    monkey = "alone"
    task = "first"

    if task == "first":
        trials = np.arange(1, 11)
    else:
        trials = np.arange(11, 21)

    drift_off = []
    rad_off = []
    diff_off = []

    drift_on = []
    rad_on = []
    diff_on = []

    for trial in trials:

        try:
            rad0, drift0, diff0 = get_drift_diff_monkey(
                monkey, "off", trial, THRESH, CUT_OFF, IF_CORRECT
            )

            rad_off.append(rad0)
            drift_off.append(drift0)
            diff_off.append(diff0)

            rad1, drift1, diff1 = get_drift_diff_monkey(
                monkey, "on", trial, THRESH, CUT_OFF, IF_CORRECT
            )

            rad_on.append(rad1)
            drift_on.append(drift1)
            diff_on.append(diff1)

        except:
            pass

    rad_off = np.hstack(rad_off)
    drift_off = np.hstack(drift_off)
    diff_off = np.hstack(diff_off)

    rad_on = np.hstack(rad_on)
    drift_on = np.hstack(drift_on)
    diff_on = np.hstack(diff_on)

    ###################
    # Drift
    ###################

    figname = task + "_exp_" + "error_hist"
    plot_hist_fit(figname, drift_off, pal[0], THRESH=THRESH, FIT=2)
    plot_hist_fit(figname, drift_on, pal[1], THRESH=THRESH, FIT=2)
    plt.xlabel("Saccadic Accuracy (°)")

    plt.vlines(np.nanmean(drift_off), 0, 0.1, color=pal[0], ls="--")
    plt.vlines(np.nanmean(drift_on), 0, 0.1, color=pal[1], ls="--")

    # plt.xlim([-THRESH, THRESH])

    plt.savefig(figname + ".svg", dpi=300)

    error_off = np.sum(np.abs(drift_off) <= 7) / drift_off.shape[0]
    error_on = np.sum(np.abs(drift_on) <= 7) / drift_on.shape[0]

    print("skewness", stat.skew(drift_off), stat.skew(drift_on))
    print("performance", error_off * 100, error_on * 100)

    print("drift", np.nanmean(drift_off), np.nanmean(drift_on))

    ###################
    # Diffusion
    ###################

    figname = task + "_exp_" + "diff_hist"
    plot_hist_fit(figname, diff_off, pal[0], THRESH=THRESH)
    plot_hist_fit(figname, diff_on, pal[1], THRESH=THRESH)
    plt.xlabel("Saccadic Precision (°)")
    # plt.xlim([-THRESH, THRESH])

    # f_value, p_value = stat.f_oneway(diff_off, diff_on)
    # f_value, p_value = f_test(diff_off, diff_on)
    value, p_value = stat.bartlett(diff_off, diff_on)
    # value, p_value = stat.levene(diff_off, diff_on, center="mean")

    print("pval", p_value)
    plt.annotate("p = {:.3f}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction")

    plt.savefig(figname + ".svg", dpi=300)

    print("bias", np.nanmean(diff_off), np.nanmean(diff_on))
    print("precision", np.nanstd(diff_off), np.nanstd(diff_on))

    figname = task + "_exp_" + "rad_hist"
    plot_hist_fit(figname, rad_off, pal[0], THRESH=THRESH)
    plot_hist_fit(figname, rad_on, pal[1], THRESH=THRESH)
    plt.xlabel("Saccadic Precision (cm)")
    plt.xlim([-THRESH, THRESH])

    plt.savefig(figname + ".svg", dpi=300)
