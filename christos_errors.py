import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import pandas as pd
import statsmodels.api as sm

from bootstrap import my_boots_compare, my_perm_test

from bootstrapped.stats_functions import mean, std

from utils import pal, get_drift_diff_monkey, plot_hist_fit

from glm import glm_abs_error


def statistic(x, y, axis):
    return np.mean(x**2, axis=axis) * x.shape[-1] / y.shape[-1] - np.mean(
        y**2, axis=axis
    )


def dispersion(x):
    return np.mean(x[0] ** 2) + np.mean(x[1] ** 2)


if __name__ == "__main__":

    THRESH = 180

    IF_CORRECT = False
    # CUT_OFF = [0, 45, 90, 180, np.nan]
    CUT_OFF = [0, 45, 90, 180, np.nan]

    monkey = "alone"
    task = "all"

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

    dum = 0
    for trial in trials:

        try:
            # if 0 == 0:
            rad0, drift0, diff0, df0 = get_drift_diff_monkey(
                monkey, "off", trial, THRESH, CUT_OFF, IF_CORRECT
            )

            rad_off.append(rad0)
            drift_off.append(drift0)
            diff_off.append(diff0)

            rad1, drift1, diff1, df1 = get_drift_diff_monkey(
                monkey, "on", trial, THRESH, CUT_OFF, IF_CORRECT
            )

            if dum == 0:
                df = pd.concat((df0, df1))
                dum = 1
            else:
                df = pd.concat((df, df0, df1))

            rad_on.append(rad1)
            drift_on.append(drift1)
            diff_on.append(diff1)

        except:
            pass

    print(df.head())
    # formula = "diff ~ monkey + trial + NB + monkey*trial + monkey*NB + trial*NB + monkey*trial*NB"
    formula = "drift ~ NB + trial + NB * trial"
    # formula = "drift ~ NB + trial + task + NB * trial + NB * task"
    # formula = "drift ~ NB + task + NB * task"

    # formula = "diff ~ NB + C(angle) + NB * C(angle)"
    # formula = "drift ~ NB + C(angle) + NB * C(angle)"

    # Fit the generalized linear model with Gaussian family and identity link
    model = sm.formula.glm(
        formula=formula, data=df.dropna(), family=sm.families.Gaussian()
    ).fit()

    # Print the summary of the model
    print(model.summary())

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
    print("mse", np.nanmean(diff_off**2), np.nanmean(diff_on**2))

    ###################
    # Diffusion
    ###################

    figname = task + "_exp_" + correct + "_diff_hist"
    plot_hist_fit(figname, diff_off, pal[0], THRESH=THRESH)
    plot_hist_fit(figname, diff_on, pal[1], THRESH=THRESH)
    plt.xlabel("Saccadic Precision (°)")
    # plt.xlim([-THRESH, THRESH])

    n_samples = 1000
    p_value = my_boots_compare(
        np.abs(diff_off),
        np.abs(diff_on),
        n_samples,
        statfunc=np.mean,
        alternative="one-sided",
        scale_test_by=len(diff_off) / len(diff_on),
    )

    print("bootstrapped p-value: {:.3f}".format(p_value))
    plt.annotate("p = {:.3f}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction")

    perm_test = stat.permutation_test(
        (np.abs(diff_off), np.abs(diff_on)),
        statistic,
        n_resamples=n_samples,
        alternative="greater",
    )

    print("permutation p-value: {:.3f}".format(perm_test.pvalue))
    if perm_test.pvalue < 0.001:
        plt.annotate(
            "p = {:.2e}".format(perm_test.pvalue),
            xy=(0.65, 0.9),
            xycoords="axes fraction",
        )
    else:
        plt.annotate(
            "p = {:.3f}".format(perm_test.pvalue),
            xy=(0.65, 0.9),
            xycoords="axes fraction",
        )

    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + correct + "_X_hist"
    plot_hist_fit(figname, rad_off[0], pal[0], THRESH=THRESH)
    plot_hist_fit(figname, rad_on[0], pal[1], THRESH=THRESH)
    plt.xlabel("X Precision")
    plt.xlim([-THRESH / 15, THRESH / 15])
    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + correct + "_Y_hist"
    plot_hist_fit(figname, rad_off[1], pal[0], THRESH=THRESH)
    plot_hist_fit(figname, rad_on[1], pal[1], THRESH=THRESH)
    plt.xlabel("Y Precision")
    plt.xlim([-THRESH / 15, THRESH / 15])
    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + correct + "_xy_scatter"
    plt.figure(figname)
    plt.scatter(rad_off[0], rad_off[1], color=pal[0], alpha=0.25)
    plt.scatter(rad_on[0], rad_on[1], color=pal[1], alpha=0.25)
    plt.xlim([-THRESH / 15, THRESH / 15])
    plt.xlabel("X Precision")
    plt.ylabel("Y Precision")
    plt.xlim([-THRESH / 15, THRESH / 15])

    cloud_off = np.vstack((rad_off[0], rad_off[1]))
    cloud_on = np.vstack((rad_on[0], rad_on[1]))

    # p_value = my_boots_compare(
    #     np.abs(np.hstack(cloud_off)),
    #     np.abs(np.hstack(cloud_on)),
    #     n_samples,
    #     statfunc=np.mean,
    #     alternative="one-sided",
    #     scale_test_by=len(np.hstack(cloud_off)) / len(np.hstack(cloud_on)),
    # )

    n_samples = 1000
    p_value = my_perm_test(
        cloud_off,
        cloud_on,
        n_samples,
        statfunc=dispersion,
        alternative="greater",
    )

    # print results
    print("perm_test p-value: {:.3f}".format(p_value))
    if p_value < 0.001:
        plt.annotate(
            "p = {:.2e}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction"
        )
    else:
        plt.annotate(
            "p = {:.3f}".format(p_value), xy=(0.05, 0.9), xycoords="axes fraction"
        )

    plt.savefig(figname + ".svg", dpi=300)
