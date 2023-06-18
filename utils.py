import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import statsmodels.api as sm

from bootstrap import my_boots_ci

# from bootstrap import my_boots_compare, my_perm_test

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7.5
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


def gaussian_fit(X, THRESH, pal):
    bins_fit = np.linspace(-THRESH, THRESH, 1000)
    mu_, sigma_ = stat.norm.fit(X)
    fit_ = stat.norm.pdf(bins_fit, mu_, sigma_)
    plt.plot(bins_fit, fit_, color=pal, lw=5)


def raw_data_to_df(THRESH=30):

    # NB OFF
    # monkey 0
    df0 = pd.read_excel(
        "./data/StimulationEyeEndPoint2.xlsx",
        engine="openpyxl",
        sheet_name="Inf_GruNoStim",
    )
    df0 = classToStim(df0)

    df0["monkey"] = np.zeros(df0.shape[0])
    df0["NB"] = np.zeros(df0.shape[0])

    df1 = pd.read_excel(
        "./data/StimulationEyeEndPoint2.xlsx",
        engine="openpyxl",
        sheet_name="Inf_HecNoStim",
    )
    df1 = classToStim(df1)

    df1["monkey"] = np.ones(df1.shape[0])
    df1["NB"] = np.zeros(df1.shape[0])

    # NB ON
    # monkey 0
    df2 = pd.read_excel(
        "./data/StimulationEyeEndPoint2.xlsx",
        engine="openpyxl",
        sheet_name="Inf_GruStim",
    )
    df2 = classToStim(df2)

    df2["monkey"] = np.zeros(df2.shape[0])
    df2["NB"] = np.ones(df2.shape[0])

    # monkey 1
    df3 = pd.read_excel(
        "./data/StimulationEyeEndPoint2.xlsx",
        engine="openpyxl",
        sheet_name="Inf_HecStim",
    )
    df3 = classToStim(df3)

    df3["monkey"] = np.ones(df3.shape[0])
    df3["NB"] = np.ones(df3.shape[0])

    df = pd.concat([df0, df1, df2, df3], ignore_index=True)
    df = df.drop(["filename", "trial", "endPointTurn"], axis=1)
    df = df[df.latency != 0]
    df.latency = df.latency.abs()

    df = df.rename(columns={"FirstStiX": "X1"})
    df = df.rename(columns={"FirstStiY": "Y1"})

    df = df.rename(columns={"SecondStiX": "X2"})
    df = df.rename(columns={"SecondStiY": "Y2"})

    df = df.rename(columns={"endpointX": "X"})
    df = df.rename(columns={"endpointY": "Y"})

    _, df["theta_S1"] = carteToPolar(df["X1"], df["Y1"])
    _, df["theta_S2"] = carteToPolar(df["X2"], df["Y2"])
    _, df["theta"] = carteToPolar(df["X"], df["Y"])

    df["distance"] = np.abs(df["theta_S1"] - df["theta_S2"])
    df["sign"] = np.sign(df["theta_S1"] - df["theta_S2"])

    df.loc[df.theta_S1.isna(), "distance"] = -np.pi / 180
    df.loc[df.theta_S2.isna(), "distance"] = -np.pi / 180

    df["error"] = np.nan * np.ones(df.shape[0])
    df["dtheta"] = np.nan * np.ones(df.shape[0])
    df["task"] = np.nan * np.ones(df.shape[0])

    df.loc[df["class"] <= 10, "task"] = 0
    df.loc[df["class"] > 10, "task"] = 1

    # df = df[df.State_code == 7]
    # print(np.mean(df[df["class"] <= 10].task == 0))
    # print(np.mean(df[df["class"] > 10].task == 1))

    df.loc[df["class"] <= 10, "error"] = (
        df.loc[df["class"] <= 10, "theta"] - df.loc[df["class"] <= 10, "theta_S1"]
    )

    # print(df.error.iloc[0] == df.theta.iloc[0] - df.theta_S1.iloc[0])

    df.loc[df["class"] > 10, "error"] = (
        df.loc[df["class"] > 10, "theta"] - df.loc[df["class"] > 10, "theta_S2"]
    )

    # print(
    #     df[df["class"] == 12].error.iloc[0]
    #     == df[df["class"] == 12].theta.iloc[0] - df[df["class"] == 12].theta_S2.iloc[0]
    # )

    df.loc[df["error"] > np.pi, "error"] = df["error"] - 2 * np.pi
    df.loc[df["error"] <= -np.pi, "error"] = df["error"] + 2 * np.pi

    df = df[np.abs(df.error) < THRESH * np.pi / 180]

    for monkey in range(2):
        for NB in range(2):
            for session in range(1, 21):
                df_class = df[
                    (df["class"] == session)
                    & (df["monkey"] == monkey)
                    & (df["NB"] == NB)
                ]

                # class_mean = stat.circmean(
                #     df_class["error"],
                #     nan_policy="omit",
                #     axis=0,
                #     high=2 * np.pi,
                #     low=0,
                # )

                # if df_class.distance.iloc[0] != 0:
                #     class_mean = df_class[
                #         (df_class.error.abs() <= df_class.distance.iloc[0] * 0.75)
                #     ].error.mean()
                # else:
                # class_mean = df_class[
                #     (df_class.error.abs() <= 30 * np.pi / 180.0)
                # ].error.mean()

                class_mean = df_class.error.mean()

                df.loc[
                    (df["class"] == session)
                    & (df["monkey"] == monkey)
                    & (df["NB"] == NB),
                    "dtheta",
                ] = (
                    df_class["error"] - class_mean
                )

    df.loc[df["dtheta"] > np.pi, "dtheta"] = df["dtheta"] - 2 * np.pi
    df.loc[df["dtheta"] <= -np.pi, "dtheta"] = df["dtheta"] + 2 * np.pi

    df["theta_S1"] *= 180 / np.pi
    df["theta_S2"] *= 180 / np.pi
    df["theta"] *= 180 / np.pi
    df["distance"] *= 180 / np.pi
    df["error"] *= 180 / np.pi
    df["dtheta"] *= 180 / np.pi

    df.loc[df.sign != 0, "error"] *= -df.sign

    df["dtheta2"] = df["dtheta"] ** 2
    df["error2"] = df["error"] ** 2

    return df


def glm_NB_task(df, error="dtheta"):
    if error == "dtheta":
        formula = "dtheta2 ~ NB * task"
    else:
        formula = "error2 ~ NB * task"

    print(formula)
    model = sm.formula.glm(
        formula=formula, data=df.dropna(), family=sm.families.Gaussian()
    ).fit()
    print(model.summary())
    return model


def glm_NB_task_monkey(df, error="dtheta"):

    if error == "dtheta":
        formula = "dtheta2 ~ NB * task * monkey"
    else:
        formula = "error2 ~ NB * task * monkey"

    print(formula)
    model = sm.formula.glm(
        formula=formula, data=df.dropna(), family=sm.families.Gaussian()
    ).fit()
    print(model.summary())
    return model


def glm_NB_distance(df, task, error="dtheta", THRESH=180):

    df2 = df[(df.task == task) & (df.distance != -1) & (df.dtheta <= THRESH)]
    print(df2.distance.unique())
    # df2 = df[(df.task == task)]

    if error == "dtheta":
        formula = "dtheta2 ~ NB * C(distance)"
    else:
        formula = "error2 ~ NB * C(distance)"

    print(formula)
    model = sm.formula.glm(
        formula=formula, data=df2.dropna(), family=sm.families.Gaussian()
    ).fit()
    print(model.summary())
    return model


def glm_NB_distance_monkey(df, task, error="dtheta", THRESH=180):

    df2 = df[(df.task == task) & (df.distance != -1) & (df.dtheta <= THRESH)]
    # df2 = df[(df.task == task)]

    if error == "dtheta":
        formula = "dtheta2 ~ NB * C(distance) + monkey"
    else:
        formula = "error2 ~ NB * C(distance)"

    print(formula)
    model = sm.formula.glm(
        formula=formula, data=df2.dropna(), family=sm.families.Gaussian()
    ).fit()
    print(model.summary())
    return model


def plot_error(df, task, dist, THRESH=30, bins="auto"):

    dft = df[np.abs(df.error) <= THRESH]

    if task == "first":
        idx = dft["class"] <= 10
    elif task == "sec":
        idx = dft["class"] > 10
    elif task == "all":
        idx = True
    elif task in range(1, 21):
        idx = dft["class"] == task

    if dist in [0, 45, 90, 180]:
        idx1 = dft["distance"] == dist
    else:
        idx1 = True

    plt.hist(
        dft[(idx) & (idx1) & (dft.NB == 0)].error,
        bins=bins,
        density=True,
        histtype="step",
    )
    plt.hist(
        dft[(idx) & (idx1) & (dft.NB == 1)].error,
        bins=bins,
        density=True,
        histtype="step",
    )

    plt.xlabel("Error (°)")
    plt.ylabel("Density")


def plot_dtheta_distance(df, task, THRESH=30):

    dft = df[np.abs(df.dtheta) <= THRESH]

    if task == "first":
        idx = dft.task == 0
    elif task == "sec":
        idx = dft.task == 1
    elif task == "all":
        idx = True

    std_off, std_on = [], []
    ci_off, ci_on = [], []

    dist_list = np.array([45, 90, 180])

    for distance in dist_list:
        idx1 = dft["distance"] == distance

        df_off = dft[(idx) & (idx1) & (dft.NB == 0)].dtheta
        df_on = dft[(idx) & (idx1) & (dft.NB == 1)].dtheta

        std_off.append(np.nanstd(df_off))
        std_on.append(np.nanstd(df_on))

        ci_off.append(my_boots_ci(df_off, statfunc=np.nanstd))
        ci_on.append(my_boots_ci(df_on, statfunc=np.nanstd))

    dist_list[-1] = 135

    figname = "dtheta_distance_" + task
    plt.figure(figname)
    plt.plot(dist_list, std_off, "-o", color=pal[0])
    plt.plot(dist_list + 5, std_on, "-o", color=pal[1])
    plt.xticks([45, 90, 135], [45, 90, 180])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Precision Bias (°)")

    plt.errorbar(dist_list, std_off, yerr=np.array(ci_off).T, color=pal[0])
    plt.errorbar(dist_list + 5, std_on, yerr=np.array(ci_on).T, color=pal[1])
    plt.savefig(figname + ".svg", dpi=300)


def plot_error_distance(df, task, THRESH=30):

    dft = df[np.abs(df.dtheta) <= THRESH]

    if task == "first":
        idx = dft.task == 0
    elif task == "sec":
        idx = dft.task == 1
    elif task == "all":
        idx = True

    std_off, std_on = [], []
    ci_off, ci_on = [], []

    dist_list = np.array([45, 90, 180])

    for distance in dist_list:
        idx1 = dft["distance"] == distance

        df_off = dft[(idx) & (idx1) & (dft.NB == 0)].error ** 2
        df_on = dft[(idx) & (idx1) & (dft.NB == 1)].error ** 2

        std_off.append(np.nanmean(df_off))
        std_on.append(np.nanmean(df_on))

        ci_off.append(my_boots_ci(df_off, statfunc=np.nanmean))
        ci_on.append(my_boots_ci(df_on, statfunc=np.nanmean))

    dist_list[-1] = 135

    figname = "error_distance_" + task
    plt.figure(figname)
    plt.plot(dist_list, std_off, "-o", color=pal[0])
    plt.plot(dist_list + 5, std_on, "-o", color=pal[1])
    plt.xticks([45, 90, 135], [45, 90, 180])
    plt.xlabel("Distance btw Targets (°)")
    plt.ylabel("Accuracy Bias (°)")

    plt.errorbar(dist_list, std_off, yerr=np.array(ci_off).T, color=pal[0])
    plt.errorbar(dist_list + 5, std_on, yerr=np.array(ci_on).T, color=pal[1])
    plt.savefig(figname + ".svg", dpi=300)


def plot_dtheta(df, task, dist, THRESH=30, bins="auto"):

    dft = df[np.abs(df.dtheta) <= THRESH]

    if task == "first":
        idx = dft.task == 0
    elif task == "sec":
        idx = dft.task == 1
    elif task == "all":
        idx = True
    elif task in range(1, 21):
        idx = dft["class"] == task

    if dist in [-1, 0, 45, 90, 180]:
        idx1 = dft["distance"] == dist
    else:
        idx1 = True

    plt.hist(
        dft[(idx) & (idx1) & (dft.NB == 0)].dtheta,
        bins=bins,
        density=True,
        histtype="step",
        color=pal[0],
        lw=5,
        alpha=0.5,
    )

    gaussian_fit(dft[(idx) & (idx1) & (dft.NB == 0)].dtheta, THRESH, pal[0])

    plt.hist(
        dft[(idx) & (idx1) & (dft.NB == 1)].dtheta,
        bins=bins,
        density=True,
        histtype="step",
        color=pal[1],
        lw=5,
        alpha=0.5,
    )

    gaussian_fit(dft[(idx) & (idx1) & (dft.NB == 1)].dtheta, THRESH, pal[1])

    plt.xlabel("Corrected Error (°)")
    plt.ylabel("Density")


def plot_latency(df, task, THRESH=30):

    dft = df[np.abs(df.dtheta) <= THRESH]

    if task == "first":
        idx = dft.task == 0
    elif task == "sec":
        idx = dft.task == 1
    elif task == "all":
        idx = True
    else:
        idx = dft["class"] == task

    latency_off = []
    latency_on = []

    distance = [-1, 0, 45, 90, 180]
    for i in distance:
        latency_off.append(
            dft[(idx) & (dft.distance == i) & (dft.NB == 0)].latency.mean()
        )
        latency_on.append(
            dft[(idx) & (dft.distance == i) & (dft.NB == 1)].latency.mean()
        )

    plt.plot(distance, latency_off, "o")
    plt.plot(distance, latency_on, "o")

    plt.xlabel("Distance (°)")
    plt.ylabel("Latency (ms)")


def plot_saccade(df, task, THRESH=30):

    dft = df[np.abs(df.dtheta) <= THRESH]

    if task == "first":
        idx = dft.task == 0
    elif task == "sec":
        idx = dft.task == 1
    elif task == "all":
        idx = True
    else:
        idx = dft["class"] == task

    saccade_off = []
    saccade_on = []

    distance = [-1, 0, 45, 90, 180]
    for i in distance:
        saccade_off.append(
            dft[(idx) & (dft.distance == i) & (dft.NB == 0)].sacc_duration.mean()
        )
        saccade_on.append(
            dft[(idx) & (dft.distance == i) & (dft.NB == 1)].sacc_duration.mean()
        )

    plt.plot(distance, saccade_off, "o")
    plt.plot(distance, saccade_on, "o")

    plt.xlabel("Distance (°)")
    plt.ylabel("Saccade Duration (ms)")


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

        if len(thetas_out[i_stim]) != 0:
            try:
                # if 0 == 0:
                radius.append(radius_out[i_stim] - radius_in[i_stim])
                sgn = -np.sign(thetas_in[i_stim][0] - thetas_cue[i_stim][0])

                if np.abs(thetas_in[i_stim][0] - thetas_cue[i_stim][0]) == 0:
                    sgn = 1

                # if np.abs(thetas_in[i_stim][0] - thetas_cue[i_stim][0]) == np.pi:
                #     sgn = 1

                drift.append(
                    get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH, CUT_OFF)
                )

                diff.append(get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH))

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
            except:
                pass

    radius = np.hstack(radius)
    drift = np.hstack(drift)
    diff = np.hstack(diff)
    cov = np.hstack(cov)

    # print(
    #     "radius",
    #     radius.shape,
    #     "drift",
    #     drift.shape,
    #     "diff",
    #     diff.shape,
    #     "XY",
    #     cov.shape,
    # )

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

    # drift = drift[np.abs(drift) < THRESH]
    drift[np.abs(drift) >= THRESH] *= np.nan

    return drift


def get_diff(theta_out, theta_in, THRESH=30, radius=1):

    diff = theta_out - theta_in

    diff[diff >= np.pi] -= 2 * np.pi
    diff[diff <= -np.pi] += 2 * np.pi
    diff *= 180 / np.pi

    mean_theta = stat.circmean(diff, nan_policy="omit", axis=0, high=180, low=-180)
    diff = diff - mean_theta

    # diff = diff[np.abs(diff) < THRESH]
    diff[np.abs(diff) >= THRESH] *= np.nan

    return diff


def get_cov(theta_out, theta_in, rad_out, rad_in, sgn, THRESH=30):

    x_out = rad_out / rad_in * np.cos(theta_out)
    y_out = rad_out / rad_in * np.sin(theta_out)

    x_in = np.cos(theta_in)
    y_in = np.sin(theta_in)

    dx = sgn * (x_out - x_in)
    dy = y_out - y_in

    dx = dx - np.nanmean(dx)
    dy = dy - np.nanmean(dy)

    r, theta = carteToPolar(dx, dy)

    theta[np.abs(r) >= 0.5] *= np.nan
    r[np.abs(r) >= 0.5] *= np.nan
    res = np.array([r * np.cos(theta), r * np.sin(theta)])

    return res


# return np.hstack((r * np.cos(theta), r * np.sin(theta)))
# return theta
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


def get_task_trial(trial):

    if trial <= 10:
        return 0
    else:
        return 1


def get_distance_trial(trial):

    if trial == 3 or trial == 8:
        return 45

    if trial == 2 or trial == 7:
        return 90

    if trial == 1 or trial == 6:
        return 180

    if trial == 4 or trial == 9:
        return 0

    if trial == 5 or trial == 10:  # null condition
        return 0

    if trial == 13 or trial == 18:
        return 45

    if trial == 12 or trial == 17:
        return 90

    if trial == 11 or trial == 16:
        return 180

    if trial == 14 or trial == 19:
        return 0

    if trial == 15 or trial == 20:  # null condition
        return 0


def get_drift_diff_monkey(monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT):

    print(condition, "trial", task)

    df0 = pd.DataFrame(
        columns=["monkey", "task", "trial", "angle", "diff", "drift", "NB"]
    )
    df1 = pd.DataFrame(
        columns=["monkey", "task", "trial", "angle", "diff", "drift", "NB"]
    )

    if monkey == "alone":
        monkey = 0
        rad_0, drift_0, diff_0 = get_drift_diff(
            monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        )

        # df0["dX"] = rad_0[0]
        # df0["dY"] = rad_0[1]

        df0["drift"] = drift_0**2
        df0["diff"] = diff_0**2

        monkey = 1
        rad_1, drift_1, diff_1 = get_drift_diff(
            monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        )

        # df1["dX"] = rad_1[0]
        # df1["dY"] = rad_1[1]

        df1["drift"] = drift_1**2
        df1["diff"] = diff_1**2

        # print(df1.shape)

        df_ = pd.concat((df0, df1))

        df_["monkey"] = np.hstack((np.zeros(diff_0.shape[0]), np.ones(diff_1.shape[0])))
        df_["angle"] = get_distance_trial(task) * np.hstack(
            (np.ones(diff_0.shape[0]), np.ones(diff_1.shape[0]))
        )

        df_["task"] = get_task_trial(task) * np.hstack(
            (np.ones(diff_0.shape[0]), np.ones(diff_1.shape[0]))
        )

        df_["trial"] = task * np.hstack(
            (np.ones(diff_0.shape[0]), np.ones(diff_1.shape[0]))
        )

        if condition == "off":
            df_["NB"] = np.hstack(
                (np.zeros(diff_0.shape[0]), np.zeros(diff_1.shape[0]))
            )
        else:
            df_["NB"] = np.hstack((np.ones(diff_0.shape[0]), np.ones(diff_1.shape[0])))

        drift_monk = np.hstack((drift_0, drift_1))
        diff_monk = np.hstack((diff_0, diff_1))
        rad_monk = np.hstack((rad_0, rad_1))

    else:
        monkey = 2
        rad_monk, drift_monk, diff_monk = get_drift_diff(
            monkey, condition, task, THRESH, CUT_OFF, IF_CORRECT
        )

    return rad_monk, drift_monk, diff_monk, df_
