from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_standard_dev(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str = "num_samples",
    y: str = "test_acc",
    hue: str = "Sampling",
    style=None,
    units=None,
    ci="sd",
    legend="auto",
    palette=None,
    markers=True,
    dashes=True,
    linestyle=None,
    err_kws=None,
) -> plt.Axes:
    """Creates a lineplot from dataframe with sns.lineplot.
    For information see:
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    """
    sns.lineplot(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        ci=ci,
        # markers=True,
        marker="o",
        units=units,
        style=style,
        legend=legend,
        palette=palette,
        markers=markers,
        dashes=dashes,
        linestyle=linestyle,
        err_kws=err_kws,
    )
    return ax


def plot_pairwise_matrix(
    matrix: Dict[str, Dict[str, float]],
    title_tag: str = "Test",
    name_dict: Dict[str, str] = None,
    max_poss_ent: int = 1,
    savepath: str = None,
    show: bool = False,
):
    """Plots or saves pairwise penalty matrix (PPM).
    Ech row i indicates the number of settings in which algorithm i beats other algorithms
    and each column j indicates the number of settings in which algorithm j is beaten by another algorithm.

    Args:
        matrix (Dict[str, Dict[str, float]]): PPM matrix.
        title_tag (str, optional): Title of Figure. Defaults to "Test".
        name_dict (Dict[str, str], optional): {name_in_matrix: name_in_plot}. Defaults to None.
        max_poss_ent (int, optional): Maximal value obtainable, equal to #AL experimentsß. Defaults to 1.
        savepath (str, optional): If not None, save to path. Defaults to None.
        show (bool, optional): If True, show plot. Defaults to False.
    """
    algs = [alg for alg in matrix.keys()]
    if name_dict is None:
        name_dict = {alg: alg for alg in algs}

    matPlot = np.zeros((len(matrix), len(matrix)))
    for c, a1 in enumerate(algs):
        for r, a2 in enumerate(algs):
            matPlot[r][c] = matrix[a1][a2]

    col_avg = matPlot.sum(axis=0) / (matPlot.shape[0] - 1)

    matPlot = np.round(matPlot, 2)
    col_avg = np.round(col_avg, 2)
    min_e = matPlot.min()
    max_e = matPlot.max()

    plt.rcParams["axes.grid"] = False
    fig, ax = plt.subplots()

    ax.tick_params(axis="both", which="both", length=0)
    im = ax.matshow(matPlot, cmap="viridis", vmin=min_e, vmax=max_e)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xticklabels([0] + [name_dict[alg] for alg in algs], fontsize=8)
    ax.set_yticklabels([0] + [name_dict[alg] for alg in algs], rotation=0, fontsize=8)

    for i in range(len(algs)):
        for j in range(len(algs)):
            text = ax.text(
                j, i, matPlot[i, j], ha="center", va="center", color="w", fontsize=8
            )

    ax.set_title(title_tag + "({})".format(max_poss_ent))
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size=1, pad=-0.25)

    for j in range(len(algs)):
        text = ax2.text(
            j, 0, col_avg[j], ha="center", va="center", color="w", fontsize=8
        )

    im = ax2.matshow(np.array([col_avg]), cmap="viridis", vmin=min_e, vmax=max_e)
    ax2.axis("off")

    fig.subplots_adjust(right=1)
    cbar_ax = fig.add_axes([0.8, 0.18, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.outline.set_visible(False)
    if show is True:
        plt.show()
    if savepath is not None:
        plt.savefig(savepath)
    plt.rcParams["axes.grid"] = True
