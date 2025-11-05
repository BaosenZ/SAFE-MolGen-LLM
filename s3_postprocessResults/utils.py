import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns


def kde_counts_series(
    x, x_min=1.8, x_max=3.0, bin_width=0.05, bw_method="scott", grid_n=512
):
    """Return xs and count-scaled KDE: counts per `bin_width`."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return None, None
    kde = gaussian_kde(x, bw_method=bw_method)
    xs = np.linspace(x_min, x_max, grid_n)
    pdf = kde(xs)
    counts = pdf * len(x) * bin_width
    return xs, counts


def plot_kde_count(df):
    order = ["N Donor", "O Donor", "Mixed N/O"]
    palette = {"N Donor": "C4", "O Donor": "C7", "Mixed N/O": "C8"}
    BIN_W = 0.05

    fig, ax = plt.subplots(figsize=(5, 3))
    for donor in order:
        x = df.loc[df["Donor"] == donor, "Lg_Bottch"].values
        xs, ys = kde_counts_series(
            x, x_min=1.8, x_max=3.0, bin_width=BIN_W, bw_method="scott"
        )
        if xs is None:
            continue
        ax.fill_between(xs, 0, ys, alpha=0.4, color=palette[donor], label=donor)
        ax.plot(xs, ys, color=palette[donor], linewidth=1.3)

    ax.set_xlim(1.8, 3.0)
    ax.set_ylim(0, 45)
    ax.set_xlabel("log₁₀(Cₘ)", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Count", fontsize=10, fontweight="bold")
    ax.legend(title="", loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_bar2_design(data_valid, data_follow, color1, color2):
    labels = ["Design 1", "Design 2", "Design 3", "Design 4"]
    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(5, 3))
    bars_valid = ax.bar(
        x - w / 2, data_valid, width=w, label="Valid SMILES", color=color1
    )
    bars_follow = ax.bar(
        x + w / 2, data_follow, width=w, label="Focus-aligned", color=color2
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    plt.show()


def plot_hist_kde(df):
    plt.figure(figsize=(10, 3))
    order = ["N Donor", "O Donor", "Mixed N/O"]
    palette = {"N Donor": "C4", "O Donor": "C7", "Mixed N/O": "C8"}

    ax = sns.histplot(
        data=df,
        x="Lg_Bottch",
        hue="Donor",
        hue_order=order,
        palette=palette,
        bins=np.arange(1.8, 3.05, 0.05),
        element="step",
        common_norm=False,
        kde=True,
    )

    sns.despine(top=True, right=True)

    leg = ax.get_legend()
    if leg:
        leg.set_title("")
        leg.set_loc("upper left")

    plt.xlim(1.8, 3)
    plt.xlabel("Lg Complexity", fontsize=10, fontweight="bold")
    plt.ylabel("Count", fontsize=10, fontweight="bold")
    plt.xticks(fontsize=10)

    plt.tight_layout()
    plt.show()
