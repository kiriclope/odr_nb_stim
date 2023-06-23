import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.io import loadmat

from decoder import get_pipeline
from my_bootstrap import my_boots_ci

sns.set_context("poster")
sns.set_style("ticks")
plt.rc("axes.spines", top=False, right=False)

golden_ratio = (5**0.5 - 1) / 2
width = 7.5
matplotlib.rcParams["figure.figsize"] = [width, width * golden_ratio]

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


def _handle_zeros_in_scale(scale, copy=True):
    """Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def z_score(X, axis=0):
    scale = _handle_zeros_in_scale(np.nanstd(X, axis))
    return (X - np.nanmean(X, axis)) / scale


def load_file():
    file = "data/Classifiy_data_odrdistvar_allNeurons_reverse.matlab"
    data = loadmat(file)
    # print(data.keys())

    return data


def array_to_df(array_2d, columns=["trial", "neuron"]):
    # Reshape the array
    reshaped_array = array_2d.reshape(-1, 1)

    # Create the DataFrame
    dft = pd.DataFrame(reshaped_array, columns=[columns[1]])
    dft[columns[0]] = np.repeat(np.arange(1, array_2d.shape[0] + 1), array_2d.shape[1])

    dft = dft.reindex(columns=columns)

    return dft


def get_df(data):

    df = pd.DataFrame()

    cue_rates = np.vstack((data["cuerate"], data["cuerate_stim"]))
    df0 = array_to_df(cue_rates, ["trial", "cue_rate"])
    df["cue_rate"] = df0["cue_rate"]

    sample_rates = np.vstack((data["samplerate"], data["samplerate_stim"]))
    df0 = array_to_df(sample_rates, ["trial", "sample_rate"])
    df["sample_rate"] = df0["sample_rate"]

    delay1_rates = np.vstack((data["cdrate"], data["cdrate_stim"]))
    df0 = array_to_df(delay1_rates, ["trial", "delay1_rate"])
    df["trial"] = df0["trial"]
    df["delay1_rate"] = df0["delay1_rate"]

    delay2_rates = np.vstack((data["sdrate"], data["sdrate_stim"]))
    df0 = array_to_df(delay2_rates, ["trial", "delay2_rate"])
    df["delay2_rate"] = df0["delay2_rate"]

    S1 = np.vstack((data["g1"], data["g1_stim"]))
    df0 = array_to_df(S1, ["trial", "S1"])
    df["S1"] = df0["S1"]

    S2 = np.vstack((data["g2"], data["g2_stim"]))
    df0 = array_to_df(S2, ["trial", "S2"])
    df["S2"] = df0["S2"]

    task = np.vstack((data["g3"], data["g3_stim"])) - 1
    df0 = array_to_df(task, ["trial", "task"])
    df["task"] = df0["task"]

    df.loc[df.trial <= 200, "NB"] = 0
    df.loc[df.trial > 200, "NB"] = 1

    print(df.head())

    return df


def get_X_y(df, epoch="delay2_rate", stim="S1"):

    print(epoch, stim)
    # Group the dataframe by 'trial' and aggregate the 'rate' column as a list
    aggregated_df = df.groupby("trial")[epoch].agg(list).reset_index()
    # Convert the list of rates for each trial into a 2D array with 65 rows
    X = np.array(aggregated_df[epoch].tolist(), dtype=object)
    X = np.array(
        [sublist + [np.nan] * (233 - len(sublist)) for sublist in X], dtype=np.float64
    )

    # print(X.shape)
    # X = z_score(X)
    # X = (X - np.nanmean(X, axis=0)) ** 2

    aggregated_df = df.groupby("trial")[stim].agg(list).reset_index()
    # Convert the list of rates for each trial into a 2D array with 65 rows
    y = np.array(aggregated_df[stim].tolist(), dtype=object)
    y = np.array(
        [sublist + [np.nan] * (233 - len(sublist)) for sublist in y], dtype=np.float64
    )

    # y[y == 1] = 0
    # y[y == 5] = 1

    return X, y


def fit_single(n_splits, pipe, X, y):

    print(np.unique(y))
    print("X", X.shape, "y", y.shape)

    scores = []
    for i in range(X.shape[1]):
        y_neuron = y[:, i]
        X_neuron = X[:, i]

        y_neuron = y_neuron[~np.isnan(X_neuron)]
        X_neuron = X_neuron[~np.isnan(X_neuron)]

        X_neuron = X_neuron.reshape(-1, 1)

        try:
            # print("neuron", i, "0", np.sum(y_neuron == 0), "180", np.sum(y_neuron == 1))
            if np.sum(y_neuron == 0) >= n_splits and np.sum(y_neuron == 1) >= n_splits:
                pipe.fit(X_neuron, y_neuron)
                score = pipe.score(X_neuron, y_neuron)
                # print(X_neuron.shape, y_neuron.shape, score)
                scores.append(score)
        except:
            pass

    print("score", np.nanmean(scores))

    return scores


if __name__ == "__main__":

    data = load_file()
    df = get_df(data)

    # df = df.dropna()

    df0 = df[df.task == 0]
    df0 = df0[df0.S1 != -1]

    df1 = df[df.task == 1]
    df1 = df1[df1.S2 != -1]

    df1 = df1[df1.S2 != 2]
    df1 = df1[df1.S2 != 3]
    df1 = df1[df1.S2 != 4]

    n_splits = -1
    pipe = get_pipeline(n_splits, penalty="l2", scoring="accuracy")

    epochs = ["cue_rate", "delay1_rate", "sample_rate", "delay2_rate"]
    stim = "S2"

    mean_scores = []
    cis = []

    for epoch in epochs:

        df_off = df1[(df1.NB == 0)]
        X_off, y_off = get_X_y(df_off, epoch, stim)

        # df_off = df0[(df0.NB == 0)]
        # X0_off, y0_off = get_X_y(df_off, epoch, stim)

        # df1_off = df1[(df1.NB == 0)]
        # X1_off, y1_off = get_X_y(df1_off, epoch, "S2")

        # # print(X0_off.shape, X1_off.shape)

        # X_off = np.vstack((X0_off, X1_off))
        # y_off = np.vstack((y0_off, y1_off))

        scores_off = fit_single(n_splits, pipe, X_off, y_off)
        ci_off = my_boots_ci(scores_off, np.nanmean)

        df_on = df1[(df1.NB == 1)]
        X_on, y_on = get_X_y(df_on, epoch, stim)

        # df0_on = df0[(df0.NB == 1)]
        # X0_on, y0_on = get_X_y(df0_on, epoch, stim)

        # df1_on = df1[(df1.NB == 1)]
        # X1_on, y1_on = get_X_y(df1_on, epoch, "S2")

        # print(X0_on.shape, X1_on.shape)

        # X_on = np.vstack((X0_on, X1_on))
        # y_on = np.vstack((y0_on, y1_on))

        scores_on = fit_single(n_splits, pipe, X_on, y_on)
        ci_on = my_boots_ci(scores_on, np.nanmean)

        print("epoch", epoch, "scores", np.nanmean(scores_off), np.nanmean(scores_on))

        mean_score = np.vstack((np.nanmean(scores_off), np.nanmean(scores_on)))[:, 0]
        ci = np.vstack((ci_off, ci_on))

        print(mean_score.shape, ci.shape)

        mean_scores.append(mean_score)
        cis.append(ci)

    mean_scores = np.array(mean_scores)
    cis = np.array(cis)

    plt.figure("acc_var")
    plt.plot(mean_scores[:, 0], color=pal[0])
    plt.plot(mean_scores[:, 1], color=pal[1])

    plt.errorbar(
        np.arange(len(epochs)), mean_scores[:, 0], yerr=cis[:, 0].T, color=pal[0]
    )
    plt.errorbar(
        np.arange(len(epochs)), mean_scores[:, 1], yerr=cis[:, 1].T, color=pal[1]
    )

    plt.xticks(np.arange(len(epochs)), ["Sample 1", "Delay 1", "Sample 2", "Delay 2"])

    plt.ylabel("AUC score")
    plt.xlabel("epoch")
