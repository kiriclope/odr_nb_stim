import numpy as np
import matplotlib.pyplot as plt


def plot_coefs(model):
    coef_names = [
        "NB:C(distance)[T.45.0]",
        "NB:C(distance)[T.90.0]",
        "NB:C(distance)[T.180.0]",
    ]
    # coef_names = ["NB:C(angle)[T.45.0]", "NB:C(angle)[T.90.0]", "NB:C(angle)[T.180.0]"]
    trial_levels = [45, 90, 135]
    # colors = ["r", "b", "g"]

    coef_vals = [model.params[name] for name in coef_names]
    se_vals = [model.bse[name] for name in coef_names]

    fig, ax = plt.subplots()
    ax.errorbar(trial_levels, coef_vals, yerr=1.96 * np.array(se_vals), fmt="o")
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_xlabel("Distance btw Targets (°)")
    ax.set_ylabel("Interaction NB-distance")
    ax.set_xticks(trial_levels, [45, 90, 180])
