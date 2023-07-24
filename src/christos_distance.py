import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

from bootstrap import my_boots_ci
from utils import *


def fit_gauss(X):
    mu_, sigma_ = stat.norm.fit(X)
    return sigma_


def fit_lognorm(X):
    shape, loc, scale = stat.lognorm.fit(X)
    return loc


if __name__ == "__main__":

    THRESH = 30
    task = "first"
    IF_CORRECT = False

    cut_offs = np.array([45, 90, 180])

    drift_D_off = []
    diff_D_off = []

    ci_drift_off = []
    ci_diff_off = []

    drift_D_on = []
    diff_D_on = []

    ci_drift_on = []
    ci_diff_on = []

    for i_cut in range(len(cut_offs)):

        CUT_OFF = [cut_offs[i_cut]]

        condition = "off"

        # monkey = 0
        # rad_0, drift_0, diff_0 = get_drift_diff(
        #     monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        # )

        # monkey = 1
        # rad_1, drift_1, diff_1 = get_drift_diff(
        #     monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        # )

        # drift_off = np.hstack((drift_0, drift_1))
        # diff_off = np.hstack((diff_0, diff_1))
        # rad_off = np.hstack((rad_0, rad_1))

        monkey = 2
        rad_off, drift_off, diff_off = get_drift_diff(monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT)

        # diff_off = drift_off - np.nanmean(drift_off[np.abs(drift_off) < THRESH])

        # drift_off = drift_off[drift_off!=0]
        # diff_off = drift_off - np.nanmean(drift_off)
        # diff_off = diff_off[np.abs(diff_off)<THRESH]

        print(
            "distance",
            cut_offs[i_cut],
            "drift_off",
            drift_off.shape,
            "diff_off",
            diff_off.shape,
        )

        # _, ci_off = my_boots_ci(drift_off, n_samples=10000, statfunc=np.nanmean)
        # _, ci_off = my_boots_ci(np.abs(drift_off), n_samples=10000, statfunc=np.nanmean)
        _, ci_off = my_boots_ci(
            drift_off, n_samples=10000, statfunc=lambda x: np.sqrt(np.nanmean(x**2))
        )

        # _, ci_off = my_boots_ci(drift_off, n_samples=10000, statfunc=lambda x: np.sqrt(np.nanmean(x**2)))

        _, ci_off_2 = my_boots_ci(diff_off, n_samples=10000, statfunc=np.nanstd)
        # _, ci_off_2 = my_boots_ci(diff_off, n_samples=10000, statfunc=fit_gauss)

        # mean_drift = np.nanmean(drift_off)
        # mean_drift = np.nanmean(np.abs(drift_off))
        mean_drift = np.sqrt(np.nanmean(drift_off**2))

        var_diff = np.nanstd(diff_off)
        # var_diff = fit_gauss(diff_off)

        drift_D_off.append(mean_drift)
        diff_D_off.append(var_diff)

        ci_drift_off.append(ci_off[0])
        ci_diff_off.append(ci_off_2[0])

        condition = "on"
        # monkey = 0
        # rad_0, drift_0, diff_0 = get_drift_diff(
        #     monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        # )
        # monkey = 1
        # rad_1, drift_1, diff_1 = get_drift_diff(
        #     monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        # )

        # drift_on = np.hstack((drift_0, drift_1))
        # diff_on = np.hstack((diff_0, diff_1))
        # rad_on = np.hstack((rad_0, rad_1))

        monkey = 2
        rad_on, drift_on, diff_on = get_drift_diff(monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT)

        # diff_on = drift_on - np.nanmean(drift_on)
        # drift_on = drift_on[drift_on!=0]
        # diff_on = drift_on - np.nanmean(drift_on)
        # diff_on = diff_on[np.abs(diff_on)<THRESH]

        print(
            "distance",
            cut_offs[i_cut],
            "drift_on",
            drift_on.shape,
            "diff_on",
            diff_on.shape,
        )

        # _, ci_on = my_boots_ci(drift_on, n_samples=10000, statfunc=np.nanmean)
        # _, ci_on = my_boots_ci(np.abs(drift_on), n_samples=10000, statfunc=np.nanmean)
        _, ci_on = my_boots_ci(
            drift_on, n_samples=10000, statfunc=lambda x: np.sqrt(np.nanmean(x**2))
        )

        _, ci_on_2 = my_boots_ci(diff_on, n_samples=10000, statfunc=np.nanstd)
        # _, ci_on_2 = my_boots_ci(diff_on, n_samples=10000, statfunc=fit_gauss)

        print(
            "distance",
            cut_offs[i_cut],
            "drift_on",
            drift_on.shape,
            "diff_on",
            diff_on.shape,
        )

        # mean_drift = np.nanmean(drift_on)
        # mean_drift = np.nanmean(np.abs(drift_on))
        mean_drift = np.sqrt(np.nanmean(drift_off**2))

        var_diff = np.nanstd(diff_on)
        # var_diff = fit_gauss(diff_on)

        drift_D_on.append(mean_drift)
        diff_D_on.append(var_diff)

        ci_drift_on.append(ci_on[0])
        ci_diff_on.append(ci_on_2[0])

    diff_D_off = np.array(diff_D_off)
    drift_D_off = np.array(drift_D_off)

    diff_D_on = np.array(diff_D_on)
    drift_D_on = np.array(drift_D_on)

    cut_offs[-1] = 135

    figname = "drift_distance_" + task
    plt.figure(figname)
    plt.plot(cut_offs, drift_D_off, "-o", color=pal[0])
    plt.plot(cut_offs + 5, drift_D_on, "-o", color=pal[1])
    plt.xticks([45, 90, 135], [45, 90, 180])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Accuracy Bias (°)")

    plt.errorbar(cut_offs, drift_D_off, yerr=np.array(ci_drift_off).T, color=pal[0])
    plt.errorbar(cut_offs + 5, drift_D_on, yerr=np.array(ci_drift_on).T, color=pal[1])
    plt.savefig(figname + ".svg", dpi=300)

    figname = "diff_distance_" + task
    plt.figure(figname)
    plt.plot(cut_offs, diff_D_off, "-o", color=pal[0])
    plt.plot(cut_offs + 5, diff_D_on, "-o", color=pal[1])
    plt.xticks([45, 90, 135], [45, 90, 180])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Precision Bias (°)")

    plt.errorbar(cut_offs, diff_D_off, yerr=np.array(ci_diff_off).T, color=pal[0])
    plt.errorbar(cut_offs + 5, diff_D_on, yerr=np.array(ci_diff_on).T, color=pal[1])
    plt.savefig(figname + ".svg", dpi=300)

    figname = "error_distance_" + task
    plt.figure(figname)
    plt.plot(cut_offs, diff_D_off + drift_D_off, "-o", color=pal[0])
    plt.plot(cut_offs + 5, diff_D_on + drift_D_on, "-o", color=pal[1])
    plt.xticks([45, 90, 135], [45, 90, 180])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Total Bias (°)")
    plt.savefig(figname + ".svg", dpi=300)
