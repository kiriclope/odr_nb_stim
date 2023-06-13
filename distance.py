import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

from utils import pal, get_drift_diff_monkey
from bootstrapped.bootstrap import bootstrap
from bootstrapped.stats_functions import mean, std

# from bootstrap import my_boots_ci


def drift_func(x, axis=0):
    return np.sqrt(np.nanmean(np.array(x) ** 2, axis))


def fit_gauss(X):
    mu_, sigma_ = stat.norm.fit(X)
    return sigma_


def fit_lognorm(X):
    shape, loc, scale = stat.lognorm.fit(X)
    return loc


if __name__ == "__main__":

    THRESH = 30
    IF_CORRECT = False
    cut_offs = np.array([45, 90, 180])
    n_samples = 10000

    monkey = "alone"
    task = "first"

    if task == "first":
        trials = np.arange(1, 11)
    elif task == "sec":
        trials = np.arange(11, 21)
    else:
        trials = np.arange(1, 21)

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

        drift_off = []
        rad_off = []
        diff_off = []

        drift_on = []
        rad_on = []
        diff_on = []

        for trial in trials:

            try:
                rad, drift, diff = get_drift_diff_monkey(
                    monkey, "off", trial, THRESH, CUT_OFF, IF_CORRECT
                )

                rad_off.append(rad)
                drift_off.append(drift)
                diff_off.append(diff)

                rad, drift, diff = get_drift_diff_monkey(
                    monkey, "on", trial, THRESH, CUT_OFF, IF_CORRECT
                )

                rad_on.append(rad)
                drift_on.append(drift)
                diff_on.append(diff)

            except:
                pass

        rad_off = np.hstack(rad_off)
        drift_off = np.hstack(drift_off)
        diff_off = np.hstack(diff_off)

        rad_on = np.hstack(rad_on)
        drift_on = np.hstack(drift_on)
        diff_on = np.hstack(diff_on)

        # _, ci_off = my_boots_ci(
        #     drift_off, n_samples=10000, statfunc=lambda x: np.sqrt(np.nanmean(x**2))
        # )

        # boots_off = bootstrap(
        #     drift_off,
        #     stat_func=mean,
        #     alpha=0.05,
        #     num_iterations=n_samples,
        # )
        # ci_off = [boots_off.lower_bound, boots_off.upper_bound]

        # boots_off_2 = bootstrap(
        #     diff_off, stat_func=std, alpha=0.05, num_iterations=n_samples
        # )
        # ci_off_2 = [boots_off_2.lower_bound, boots_off_2.upper_bound]

        boots_off = stat.bootstrap(
            (drift_off,),
            statistic=drift_func,
            n_resamples=n_samples,
            method="Bca",
            confidence_level=0.95,
        )
        ci_off = boots_off.confidence_interval

        boots_off_2 = stat.bootstrap(
            (np.abs(diff_off),),
            statistic=np.nanmean,
            n_resamples=n_samples,
            method="Bca",
            confidence_level=0.95,
        )
        ci_off_2 = boots_off_2.confidence_interval

        # _, ci_off_2 = my_boots_ci(diff_off, n_samples=10000, statfunc=np.nanstd)
        # _, ci_off_2 = my_boots_ci(diff_off, n_samples=10000, statfunc=fit_gauss)

        mean_drift = np.sqrt(np.nanmean(np.array(drift_off) ** 2))
        var_diff = np.nanmean(np.abs(diff_off))

        # var_diff = np.nanstd(diff_off)
        # var_diff = fit_gauss(diff_off)

        drift_D_off.append(mean_drift)
        diff_D_off.append(var_diff)

        ci_drift_off.append(ci_off)
        ci_diff_off.append(ci_off_2)

        # boots_on = bootstrap(
        #     drift_on,
        #     stat_func=mean,
        #     alpha=0.05,
        #     num_iterations=n_samples,
        # )
        # ci_on = [boots_on.lower_bound, boots_on.upper_bound]

        # boots_on_2 = bootstrap(
        #     diff_on, stat_func=std, alpha=0.05, num_iterations=n_samples
        # )
        # ci_on_2 = [boots_on_2.lower_bound, boots_on_2.upper_bound]

        boots_on = stat.bootstrap(
            (drift_on,),
            statistic=drift_func,
            n_resamples=n_samples,
            method="Bca",
            confidence_level=0.95,
        )
        ci_on = boots_on.confidence_interval

        boots_on_2 = stat.bootstrap(
            (np.abs(diff_on),),
            statistic=np.nanmean,
            n_resamples=n_samples,
            method="Bca",
            confidence_level=0.95,
        )
        ci_on_2 = boots_on_2.confidence_interval

        # _, ci_on = my_boots_ci(
        #     drift_on, n_samples=10000, statfunc=lambda x: np.sqrt(np.nanmean(x**2))
        # )
        # _, ci_on_2 = my_boots_ci(diff_on, n_samples=10000, statfunc=np.nanstd)
        # _, ci_on_2 = my_boots_ci(diff_on, n_samples=10000, statfunc=fit_gauss)

        # mean_drift = np.nanmean(drift_on)
        # mean_drift = np.nanmean(np.abs(drift_on))
        mean_drift = np.sqrt(np.nanmean(np.array(drift_on) ** 2))
        var_diff = np.nanmean(np.abs(diff_on))

        # var_diff = np.nanstd(diff_on)
        # var_diff = fit_gauss(diff_on)

        drift_D_on.append(mean_drift)
        diff_D_on.append(var_diff)

        ci_drift_on.append(ci_on)
        ci_diff_on.append(ci_on_2)

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
