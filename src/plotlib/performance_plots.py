from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def plot_standard_dev(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str = "num_samples",
    y: str = "test_acc",
    hue: str = "Sampling",
    style: Optional[str] = None,
    units: Optional[str] = None,
    ci: Optional[str] = "sd",
) -> plt.Axes:
    """Creates a lineplot from dataframe with sns.lineplot.
    For information see:
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    Returns:
        plt.Axes: _description_
    """
    sns.lineplot(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        ci=ci,
        markers=True,
        marker="o",
        dashes=True,
        units=units,
        style=style,
    )
    return ax
