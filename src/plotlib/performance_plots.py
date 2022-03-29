import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def plot_standard_dev(
    ax: plt.axis, data: pd.DataFrame, x="num_samples", y="test_acc", hue="Sampling"
):
    sns.lineplot(
        ax=ax,
        data=data,
        x=x,
        y=y,
        hue=hue,
        ci="sd",
        markers=True,
        marker="o",
        dashes=False,
    )
    return ax
