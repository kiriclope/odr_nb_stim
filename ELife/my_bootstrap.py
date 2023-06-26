import numpy as np
from scipy.stats import bootstrap
from joblib import Parallel, delayed
from sklearn.utils import resample, shuffle

# from bootstrapped import bootstrap as bs
# from bootstrapped import compare_functions as bs_compare


def get_pvalue(stat, obs, n_samples, alternative="greater"):

    if alternative == "greater":
        p_value = np.sum(stat >= obs) / n_samples
    elif alternative == "less":
        p_value = np.sum(stat <= obs) / n_samples
    else:
        p_value = 2.0 * min(np.mean(stat >= obs), np.mean(stat <= obs))

    return p_value


def perm_test_loop(X, Y, statfunc):

    # rng = np.random.default_rng(seed=None)

    # X_perm = resample(X)
    # Y_perm = resample(Y)

    XY = shuffle(np.hstack((X, Y)))
    # rng.shuffle(XY, axis=-1)

    X_perm = XY[:, : X.shape[-1]]
    Y_perm = XY[:, X.shape[-1] :]

    # print(X_perm.shape, Y_perm.shape)

    stat = statfunc(X_perm) * X.shape[-1] / Y.shape[-1] - statfunc(Y_perm)
    # stat = statfunc(X_perm) - statfunc(Y_perm)

    return stat


def my_perm_test(X, Y, n_samples=10000, statfunc=np.mean, alternative="greater"):

    obs = statfunc(X) - statfunc(Y)

    # with pgb.tqdm_joblib(pgb.tqdm(desc="shuffle", total=n_samples)):
    stat = Parallel(n_jobs=-1)(
        delayed(perm_test_loop)(X, Y, statfunc) for _ in range(n_samples)
    )
    stat = np.array(stat)
    print(stat.shape)

    p_value = get_pvalue(stat, obs, n_samples, alternative)

    return p_value


def my_boots_compare(
    X,
    Y,
    n_samples=10000,
    statfunc=np.mean,
    alternative="two-sided",
    method="BCa",
    scale_test_by=1.0,
):

    observed_difference = statfunc(X) - statfunc(Y)

    boots_X = bootstrap(
        (X,),
        statistic=statfunc,
        n_resamples=n_samples,
        method=method,
        vectorized=True,
    )

    boots_Y = bootstrap(
        (Y,),
        statistic=statfunc,
        n_resamples=n_samples,
        method=method,
        vectorized=True,
    )

    bootstrap_differences = (
        boots_X.bootstrap_distribution * scale_test_by - boots_Y.bootstrap_distribution
    )

    # ## # use bootstrap to estimate the confidence interval using your test statistic
    # bootstrap_differences = bs.bootstrap_ab(
    #     X,
    #     Y,
    #     stat_func=mean,
    #     compare_func=bs_compare.difference,
    #     num_iterations=n_samples,
    #     scale_test_by=len(X) / len(Y),
    #     alpha=0.05,
    #     return_distribution=True,
    # )

    # calculate p-value (one-sided right-tail)
    if alternative == "one-sided":
        p_value = np.sum(bootstrap_differences >= observed_difference) / n_samples
    else:
        p_value = (
            2.0
            * min(
                np.sum(bootstrap_differences >= observed_difference),
                np.sum(bootstrap_differences <= observed_difference),
            )
            / n_samples
        )

    return p_value


def my_boots_ci(
    X, statfunc, n_samples=10000, method="BCa", alpha=0.05, vectorized=False, means=None
):

    boots_samples = bootstrap(
        (X,),
        statistic=statfunc,
        n_resamples=n_samples,
        method=method,
        confidence_level=1.0 - alpha,
        vectorized=vectorized,
    )

    # print(boots_samples)

    ci = [boots_samples.confidence_interval.low, boots_samples.confidence_interval.high]
    mean_boots = np.mean(boots_samples.bootstrap_distribution)

    # if means is None:
    ci[0] = mean_boots - ci[0]
    ci[1] = ci[1] - mean_boots
    # else:
    #     ci[0] = np.abs(means - ci[0])
    #     ci[1] = np.abs(ci[1] - means)

    print(ci)
    return ci
