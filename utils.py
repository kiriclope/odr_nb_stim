import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7.5
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


def plot_hist_fit(figname, X, pal, THRESH=30, FIT=1):

    plt.figure(figname)
    X = X[~np.isnan(X)]

    _, bins, _ = plt.hist(
        X, histtype="step", color=pal, density=1, alpha=0.5, bins="auto", lw=5
    )

    if FIT == 1:
        bins_fit = np.linspace(-THRESH, THRESH, 1000)
        mu_, sigma_ = stat.norm.fit(X)
        fit_ = stat.norm.pdf(bins_fit, mu_, sigma_)
        plt.plot(bins_fit, fit_, color=pal, lw=5)

    if FIT == 2:
        bins_fit = np.linspace(-THRESH, THRESH, 1000)
        shape, loc, scale = stat.lognorm.fit(X)
        fit_ = stat.lognorm.pdf(bins_fit, shape, loc=loc, scale=scale)
        plt.plot(bins_fit, fit_, color=pal, lw=5)

    plt.ylabel("Density")


def add_pval(pvalue):

    if pvalue < 0.001:
        plt.text(3.25, y + 0.1, "***")
    elif pvalue < 0.01:
        plt.text(3.4, y + 0.1, "**")
    elif pvalue < 0.05:
        plt.text(3.45, y + 0.1, "*")
    else:
        plt.text(3.25, y + 0.2, "n.s.")


def get_drift_diff(
    monkey, condition, task, THRESH=30, CUT_OFF=[0, 45], IF_CORRECT=True
):

    if monkey == 2:
        df = get_df(0, condition, task, IF_CORRECT)
        df2 = get_df(1, condition, task, IF_CORRECT)
        df = pd.concat([df, df2])
    else:
        df = get_df(monkey, condition, task, IF_CORRECT)

    radius = []
    drift = []
    diff = []
    cov = []

    radius_end, radius_stim, radius_cue, theta_end, theta_stim, theta_cue = get_phases(
        df
    )

    thetas_out, thetas_in, thetas_cue, radius_out, radius_in = get_theta_loc(
        theta_end,
        theta_stim,
        theta_cue,
        radius_end,
        radius_stim,
        radius_cue,
        CUT_OFF,
        task,
    )

    for i_stim in range(thetas_out.shape[0]):

        # print(thetas_out[i_stim])

        # if condition == "off":
        #     circular_plot(
        #         radius_out[i_stim] * np.cos(thetas_out[i_stim]),
        #         radius_out[i_stim] * np.sin(thetas_out[i_stim]),
        #         "responses",
        #         color=pal[0],
        #     )
        # else:
        #     circular_plot(
        #         radius_out[i_stim] * np.cos(thetas_out[i_stim]),
        #         radius_out[i_stim] * np.sin(thetas_out[i_stim]),
        #         "responses",
        #         color=pal[1],
        #     )

        if len(thetas_out[i_stim]) != 0:
            radius.append(radius_out[i_stim] / radius_in[i_stim])
            sgn = 1

            if np.abs(thetas_in[i_stim][0] - thetas_cue[i_stim][0]) != 0:
                sgn = -np.sign(thetas_in[i_stim][0] - thetas_cue[i_stim][0])

            drift.append(
                sgn * get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH, CUT_OFF)
            )

            diff.append(sgn * get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH))

            cov.append(
                get_cov(
                    thetas_out[i_stim],
                    thetas_in[i_stim],
                    radius_out[i_stim],
                    radius_in[i_stim],
                    sgn,
                    THRESH,
                )
            )
        else:
            radius.append(np.nan)
            drift.append(np.nan)
            diff.append(np.nan)

    radius = np.hstack(radius)
    drift = np.hstack(drift)
    diff = np.hstack(diff)
    cov = np.hstack(cov)

    # print("radius", radius.shape, "drift", drift.shape, "diff", diff.shape)

    return cov, drift, diff


def classToStim(df):

    first_X = [
        12,
        12,
        12,
        12,
        12,
        -12,
        -12,
        -12,
        -12,
        -12,
        12,
        12,
        12,
        12,
        np.nan,
        -12,
        -12,
        -12,
        -12,
        np.nan,
    ]
    first_Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, np.nan]

    second_X = [
        -12,
        0,
        8.48528137423857,
        12,
        np.nan,
        12,
        0,
        -8.48528137423857,
        -12,
        np.nan,
        -12,
        0,
        8.48528137423857,
        12,
        12,
        12,
        0,
        -8.48528137423857,
        -12,
        -12,
    ]
    second_Y = [
        0,
        12,
        8.48528137423857,
        0,
        np.nan,
        0,
        12,
        8.48528137423857,
        0,
        np.nan,
        0,
        12,
        8.48528137423857,
        0,
        0,
        0,
        12,
        8.48528137423857,
        0,
        0,
    ]

    for i_class in range(20):
        df.loc[df["class"] == i_class + 1, "FirstStiX"] = first_X[i_class]
        df.loc[df["class"] == i_class + 1, "FirstStiY"] = first_Y[i_class]

        df.loc[df["class"] == i_class + 1, "SecondStiX"] = second_X[i_class]
        df.loc[df["class"] == i_class + 1, "SecondStiY"] = second_Y[i_class]

    return df


def carteToPolar(x, y):
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return radius, theta


def get_theta_loc(
    theta_end,
    theta_stim,
    theta_cue,
    radius_end,
    radius_stim,
    radius_cue,
    CUT_OFF=[45, 135],
    task="first",
):

    stim = [0, np.pi, np.nan]
    cue = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, np.nan]

    thetas_out = []
    thetas_in = []
    thetas_cue = []
    rad_out = []
    rad_in = []

    for i in range(len(stim)):

        if np.isnan(stim[i]):
            idx_stim = np.where(np.isnan(theta_stim))[0].tolist()
        else:
            idx_stim = np.where(theta_stim == stim[i])[0].tolist()

        for j in range(len(cue)):
            delta = np.abs(stim[i] - cue[j]) * 180 / np.pi

            if np.isnan(cue[j]):
                idx_cue = np.where(np.isnan(theta_cue))[0].tolist()
            else:
                idx_cue = np.where(theta_cue == cue[j])[0].tolist()

            idx = list(set(idx_stim) & set(idx_cue))

            if np.isnan(stim[i]):
                theta_stim.iloc[idx] = cue[j]

            if np.isnan(cue[j]):
                theta_cue.iloc[idx] = stim[i]

            # print("delta", delta, "idx", idx)

            if (
                (delta in CUT_OFF) or np.isnan(delta) * np.isnan(CUT_OFF).any()
            ) and len(idx) > 0:

                print(
                    "delta",
                    delta,
                    "stim",
                    stim[i] * 180 / np.pi,
                    "cue",
                    cue[j] * 180 / np.pi,
                    "idx",
                    len(idx),
                )

                thetas_out.append(theta_end.iloc[idx].to_numpy())
                rad_out.append(radius_end.iloc[idx].to_numpy())

                if task <= 10:
                    thetas_in.append(theta_stim.iloc[idx].to_numpy())
                    thetas_cue.append(theta_cue.iloc[idx].to_numpy())
                    rad_in.append(radius_stim.iloc[idx].to_numpy())
                else:
                    thetas_in.append(theta_cue.iloc[idx].to_numpy())
                    thetas_cue.append(theta_stim.iloc[idx].to_numpy())
                    rad_in.append(radius_cue.iloc[idx].to_numpy())

    thetas_out = np.array(thetas_out)
    thetas_in = np.array(thetas_in)
    thetas_cue = np.array(thetas_cue)

    rad_out = np.array(rad_out)
    rad_in = np.array(rad_in)

    return thetas_out, thetas_in, thetas_cue, rad_out, rad_in


def circular_plot(x, y, figname, color="b"):
    plt.figure(figname)
    plt.plot(x, y, "x", ms=5, color=color)
    plt.plot([0], [0], "+", ms=10, color="k")
    plt.axis("off")


def correct_sessions(df, task="first", IF_CORRECT=True):
    correct_df = df

    # print("correct", IF_CORRECT)
    if IF_CORRECT:
        # print("perf", np.sum(df.State_code == 7) / df.State_code.shape[0])
        correct_df = df[(df.State_code == 7)]

    correct_df = correct_df[correct_df.latency != 0]

    if task == "first":
        correct_df = correct_df[correct_df["class"] <= 10]
    elif task == "sec":
        correct_df = correct_df[correct_df["class"] > 10]
    else:
        correct_df = correct_df[correct_df["class"] == task]

    return correct_df


def get_X_Y(df):

    X_end = df.endpointX
    Y_end = df.endpointY

    X_stim = df.FirstStiX
    Y_stim = df.FirstStiY

    X_cue = df.SecondStiX
    Y_cue = df.SecondStiY

    return X_end, Y_end, X_stim, Y_stim, X_cue, Y_cue


def get_phases(df):

    X_end = df.endpointX
    Y_end = df.endpointY

    X_stim = df.FirstStiX
    Y_stim = df.FirstStiY

    X_cue = df.SecondStiX
    Y_cue = df.SecondStiY

    radius_end, theta_end = carteToPolar(X_end, Y_end)
    # print('theta', theta_end.shape)

    radius_stim, theta_stim = carteToPolar(X_stim, Y_stim)
    # print('theta_stim', theta_end.shape)

    radius_cue, theta_cue = carteToPolar(X_cue, Y_cue)

    return radius_end, radius_stim, radius_cue, theta_end, theta_stim, theta_cue


def get_drift(theta_out, theta_in, THRESH=30, CUT_OFF=[0]):

    drift = theta_out - theta_in

    drift[drift > np.pi] -= 2 * np.pi
    drift[drift < -np.pi] += 2 * np.pi
    drift *= 180 / np.pi

    drift = drift[np.abs(drift) < THRESH]

    return drift


def get_diff(theta_out, theta_in, THRESH=30, radius=1):

    diff = (theta_out - theta_in) * radius

    diff[diff >= np.pi] -= 2 * np.pi
    diff[diff <= -np.pi] += 2 * np.pi
    diff *= 180 / np.pi

    mean_theta = stat.circmean(diff, nan_policy="omit", axis=0, high=180, low=-180)
    diff = diff - mean_theta

    diff = diff[np.abs(diff) < THRESH]

    return diff


def get_cov(theta_out, theta_in, rad_out, rad_in, sgn, THRESH=30):

    x_out = rad_out * np.cos(theta_out)
    y_out = rad_out * np.sin(theta_out)

    x_in = rad_in * np.cos(theta_in)
    y_in = rad_in * np.sin(theta_in)

    dx = sgn * (x_out - x_in)
    dy = y_out - y_in

    # dx = dx - np.nanmean(dx)
    # dy = dy - np.nanmean(dy)

    r, theta = carteToPolar(dx, dy)

    theta = theta[np.abs(r) < THRESH]
    r = r[np.abs(r) < THRESH]

    # return r * np.cos(theta) + r * np.sin(theta)
    # return np.hstack((r * np.cos(theta), r * np.sin(theta)))

    return theta


# return theta * 180 / np.pi


def get_df(monkey, condition, task, IF_CORRECT=True):

    if condition == "on":
        if monkey == 1:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_HecStim",
            )
            df = classToStim(df)
        else:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_GruStim",
            )
    else:
        if monkey == 1:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_HecNoStim",
            )
            df = classToStim(df)
        else:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_GruNoStim",
            )

    df["filename"] = df["filename"].astype("string")
    df_correct = correct_sessions(df, task, IF_CORRECT)

    return df_correct


def get_drift_diff_monkey(monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT):

    print(condition, "trial", task)

    if monkey == "alone":
        monkey = 0
        rad_0, drift_0, diff_0 = get_drift_diff(
            monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        )

        monkey = 1
        rad_1, drift_1, diff_1 = get_drift_diff(
            monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        )

        drift_monk = np.hstack((drift_0, drift_1))
        diff_monk = np.hstack((diff_0, diff_1))
        rad_monk = np.hstack((rad_0, rad_1))

    else:
        monkey = 2
        rad_monk, drift_monk, diff_monk = get_drift_diff(
            monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        )

    return rad_monk, drift_monk, diff_monk
