import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

# from bootstrap import my_boots

from bootstrapped import bootstrap as bs
from bootstrapped import compare_functions as bs_compare
from bootstrapped.stats_functions import mean, std

from utils import pal, get_drift_diff_monkey, plot_hist_fit

from glm import glm_abs_error


def total_error(X, axis=1):

    return np.sqrt(np.mean(X[0] ** 2) + np.mean(X[1] ** 2))


if __name__ == "__main__":

    THRESH = 20

    IF_CORRECT = False
    CUT_OFF = [0, 45, 90, 180, np.nan]
    # CUT_OFF = [0]

    monkey = "alone"
    task = "first"

    if task == "first":
        trials = np.arange(1, 11)
    elif task == "sec":
        trials = np.arange(11, 21)
    else:
        trials = np.arange(1, 21)

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

    if IF_CORRECT:
        correct = "correct"
    else:
        correct = ""

    ###################
    # Drift
    ###################

    figname = task + "_exp_" + correct + "_error_hist"
    plot_hist_fit(figname, drift_off, pal[0], THRESH=THRESH, FIT=0)
    plot_hist_fit(figname, drift_on, pal[1], THRESH=THRESH, FIT=0)
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

    figname = task + "_exp_" + correct + "_diff_hist"
    plot_hist_fit(figname, diff_off, pal[0], THRESH=THRESH)
    plot_hist_fit(figname, diff_on, pal[1], THRESH=THRESH)
    plt.xlabel("Saccadic Precision (°)")
    # plt.xlim([-THRESH, THRESH])

    # f_value, p_value = f_test(diff_off, diff_on)
    value, p_value = stat.levene(diff_off, diff_on, center="mean")
    print("levene", p_value)

    n_samples = 100000
    # boots_off = my_boots(diff_off, n_samples, statfunc=np.nanstd, n_jobs=-1, verbose=0)
    # boots_on = my_boots(diff_on, n_samples, statfunc=np.nanstd, n_jobs=-1, verbose=0)

    observed_difference = np.nanmean(np.abs(diff_off)) - np.nanmean(np.abs(diff_on))

    # # Calculate the difference in the means of the bootstrap samples
    # bootstrap_differences = boots_off - boots_on

    # use bootstrap to estimate the confidence interval using your test statistic
    bootstrap_differences = bs.bootstrap_ab(
        np.abs(diff_off),
        np.abs(diff_on),
        stat_func=mean,
        compare_func=bs_compare.difference,
        num_iterations=n_samples,
        scale_test_by=len(diff_off) / len(diff_on),
        alpha=0.05,
        return_distribution=True,
    )

    # calculate p-value
    # p_value = np.sum(abs(bootstrap_differences) >= abs(observed_difference)) / n_samples

    p_value = (
        2.0
        * min(
            np.sum(bootstrap_differences >= observed_difference),
            np.sum(bootstrap_differences <= observed_difference),
        )
        / n_samples
    )

    # print results
    print("bootstrapped p-value: {:.3f}".format(p_value))
    plt.annotate("p = {:.3f}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction")

    print("bias", np.nanmean(diff_off), np.nanmean(diff_on))
    print("precision", np.nanstd(diff_off), np.nanstd(diff_on))

    model = glm_abs_error(np.abs(diff_off), np.abs(diff_on))
    p_value = model.pvalues["condition"]

    print("glm, p-value: {:.3f}".format(p_value))

    p_value = stat.f.sf(
        np.var(diff_off, ddof=1) / np.var(diff_on, ddof=1),
        diff_off.shape[0] - 1,
        diff_on.shape[0] - 1,
    )

    print("f test, p-value: {:.3f}".format(p_value))

    if p_value < 0.001:
        plt.annotate(
            "p = {:.2e}".format(p_value),
            xy=(0.6, 0.9),
            xycoords="axes fraction",
        )
    else:
        plt.annotate(
            "p = {:.3f}".format(p_value),
            xy=(0.6, 0.9),
            xycoords="axes fraction",
        )

    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + "X_hist"
    plot_hist_fit(figname, rad_off[0], pal[0], THRESH=THRESH)
    plot_hist_fit(figname, rad_on[0], pal[1], THRESH=THRESH)
    plt.xlabel("X Precision")
    plt.xlim([-THRESH / 15, THRESH / 15])

    bootstrap_differences = bs.bootstrap_ab(
        np.abs(rad_off[0]),
        np.abs(rad_on[0]),
        stat_func=mean,
        compare_func=bs_compare.difference,
        num_iterations=n_samples,
        scale_test_by=len(rad_off[0]) / len(rad_on[0]),
        alpha=0.05,
        return_distribution=True,
    )

    observed_difference = np.nanmean(np.abs(rad_off[0])) - np.nanmean(np.abs(rad_on[0]))

    p_value = (
        2.0
        * min(
            np.sum(bootstrap_differences >= observed_difference),
            np.sum(bootstrap_differences <= observed_difference),
        )
        / n_samples
    )

    # print results
    print("bootstrapped p-value: {:.3f}".format(p_value))
    plt.annotate("p = {:.3f}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction")

    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + "Y_hist"
    plot_hist_fit(figname, rad_off[1], pal[0], THRESH=THRESH)
    plot_hist_fit(figname, rad_on[1], pal[1], THRESH=THRESH)
    plt.xlabel("Y Precision")
    plt.xlim([-THRESH / 15, THRESH / 15])

    bootstrap_differences = bs.bootstrap_ab(
        np.abs(rad_off[1]),
        np.abs(rad_on[1]),
        stat_func=mean,
        compare_func=bs_compare.difference,
        num_iterations=n_samples,
        scale_test_by=len(rad_off[0]) / len(rad_on[0]),
        alpha=0.05,
        return_distribution=True,
    )

    observed_difference = np.nanmean(np.abs(rad_off[1])) - np.nanmean(np.abs(rad_on[1]))

    p_value = (
        2.0
        * min(
            np.sum(bootstrap_differences >= observed_difference),
            np.sum(bootstrap_differences <= observed_difference),
        )
        / n_samples
    )

    # print results
    print("bootstrapped p-value: {:.3f}".format(p_value))
    plt.annotate("p = {:.3f}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction")

    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + "xy_scatter"
    plt.figure(figname)
    plt.scatter(rad_off[0], rad_off[1], color=pal[0], alpha=0.25)
    plt.scatter(rad_on[0], rad_on[1], color=pal[1], alpha=0.25)
    plt.xlim([-THRESH / 15, THRESH / 15])
    plt.xlabel("X Precision")
    plt.ylabel("Y Precision")

    plt.savefig(figname + ".svg", dpi=300)
