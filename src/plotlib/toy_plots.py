from typing import Dict, Tuple
import numpy as np

import matplotlib.pyplot as plt
from math import ceil

import seaborn as sns

sns.set_style()

### data helper functions


def create_2d_grid_from_data(X: np.ndarray):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    return xx, yy


## whole figure routines creating figures and axes


def fig_class_full_2d(
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    lab_train: np.ndarray,
    lab_val: np.ndarray,
    grid_lab: np.ndarray,
    grid_arrays: Tuple[np.ndarray, np.ndarray],
    pred_unlabelled: np.ndarray = None,
    pred_queries: np.ndarray = None,
):
    fig, axes = plt.subplots(1, 2, sharex="col", sharey="row")
    axes[0].set_title("Training Data")
    axes[0] = vis_class_train_2d(
        axes[0],
        pred_train,
        lab_train,
        grid_lab,
        grid_arrays,
        pred_unlabelled,
        pred_queries,
    )
    axes[1].set_title("Validation Data")
    axes[1] = vis_class_val_2d(axes[1], pred_val, lab_val, grid_lab, grid_arrays)
    return fig, axes


def fig_uncertain_full_2d(
    pred_train: np.ndarray,
    lab_train: np.ndarray,
    grid_uncertainties: Dict[str, np.ndarray],
    grid_arrays: Tuple[np.ndarray, np.ndarray],
    max_cols=4,
):
    """Visualizes multiple uncertainty grids based on grid_uncertainties.

    Args:
        pred_train (np.ndarray): predictors of training set
        lab_train (np.ndarray): labels of training set
        grid_uncertainties (Dict[str, np.ndarray]): Dict of arrays which are used for uncertainty map
        grid_arrays (Tuple[np.ndarray, np.ndarray]): Two numpy arrays (xx and yy) from np.meshgrid
        max_cols (int, optional): maximal columns of plot. Defaults to 4.

    Returns:
        Tupe(plt.figure, plt.axes): figure and axes for plots
    """
    keys = [key for key in grid_uncertainties.keys()]
    num_plots = len(keys)
    # TODO: Future - think about how to generalize this function
    num_rows = ceil(num_plots / max_cols)
    num_cols = min(num_plots, max_cols)
    fig, axs = plt.subplots(num_rows, num_cols, sharex="col", sharey="row")
    if num_cols == 1:
        axs = [axs]
    if num_rows == 1:
        axs = [axs]
    for y, ax_row in enumerate(axs):
        for x, ax in enumerate(ax_row):
            count = x + max_cols * y

            if count >= num_plots:
                continue
            key = keys[count]
            grid_unc = grid_uncertainties[key]
            ax.set_title("Uncertainty Map for {}".format(key))
            ax = vis_unc_train_2d(ax, pred_train, lab_train, grid_unc, grid_arrays)
    return fig, axs


## whole visualization routines on axes


def vis_unc_train_2d(
    ax,
    predictors,
    labels,
    grid_labels,
    grid_arrays,
):
    ax = scatter_class(ax, predictors, labels)

    xx, yy = grid_arrays
    if xx.shape != yy.shape:
        raise ValueError(
            "Object grid_arrays needs appropriate inputs with identical sizes!"
        )

    ax = contourf_contin(ax, xx, yy, grid_labels)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax


def vis_class_train_2d(
    ax,
    predictors,
    labels,
    grid_labels,
    grid_arrays,
    predictors_unlabeled=None,
    predictors_query=None,
):
    if predictors_unlabeled is not None:
        ax = scatter_unlabeled(ax, predictors_unlabeled)
    if predictors_query is not None:
        ax = scatter_query(ax, predictors_query)
    ax = scatter_class(ax, predictors, labels)

    xx, yy = grid_arrays
    if xx.shape != yy.shape:
        raise ValueError(
            "Object grid_arrays needs appropriate inputs with identical sizes!"
        )

    ax = contourf_class(ax, xx, yy, grid_labels)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax


def vis_class_val_2d(ax, predictors, labels, grid_labels, grid_arrays):
    ax = scatter_class(ax, predictors, labels)

    xx, yy = grid_arrays
    if xx.shape != yy.shape:
        raise ValueError(
            "Object grid_arrays needs appropriate inputs with identical sizes!"
        )

    ax = contourf_class(ax, xx, yy, grid_labels)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax


## simplified plotting functions on axes


def scatter_class(ax, predictors, labels):
    ax.scatter(predictors[:, 0], predictors[:, 1], c=labels, s=20, edgecolor="k")
    return ax


def scatter_query(ax, predictors, labels=None):
    ax.scatter(predictors[:, 0], predictors[:, 1], c="red", s=20, edgecolor="k")
    return ax


def scatter_unlabeled(ax, predictors, labels=None):
    ax.scatter(predictors[:, 0], predictors[:, 1], c="gray", s=15, edgecolor="k")
    return ax


def contourf_class(ax, xx, yy, labels_grid):
    ax.contourf(xx, yy, labels_grid.reshape(xx.shape), alpha=0.3)
    return ax


def contourf_contin(ax, xx, yy, prob_grid):
    ax.contourf(xx, yy, prob_grid.reshape(xx.shape), alpha=0.3, cmap="RdYlBu_r")
    return ax


def imshow_contin(ax, xx, yy, prob):
    ax.imshow(
        prob.reshape(xx.shape), interpolation="nearest", cmap="RdYlBu_r", origin="lower"
    )
    return ax


def run_example_data():
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF

    # from sklearn.model_selection import GridSearchCV #TODO: potentially experiment with this here!
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    xx, yy = create_2d_grid_from_data(X)
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    clf = DecisionTreeClassifier(max_depth=4)
    clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.8, random_state=42
    )

    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)
    pred_test = clf.predict(X_test)

    prob_train = clf.predict_proba(X_train)
    prob_val = clf.predict_proba(X_val)
    prob_test = clf.predict_proba(X_test)

    pred_grid = clf.predict(X_grid)
    prob_grid = clf.predict_proba(X_grid)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = scatter_class(ax, X_test, y_test)
    ax = contourf_class(ax, xx, yy, pred_grid)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    # ax = class_scatter(ax, X_test, y_test)
    plt.savefig("example_class.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = scatter_class(ax, X_test, y_test)
    ax = contourf_contin(ax, xx, yy, prob_grid[:, 0])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.savefig("example_prob-1.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = scatter_class(ax, X_test, y_test)
    ax = imshow_contin(ax, xx, yy, prob_grid[:, 0])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.savefig("example_prob-2.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = scatter_unlabeled(ax, X_val)
    ax = scatter_class(ax, X_test, y_test)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    plt.savefig("example_scatter_unlabeled.png")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = vis_class_train_2d(ax, X_train, y_train, pred_grid, (xx, yy), X_val)
    plt.savefig("example_train_2d.png")

    fig, axes = fig_class_full_2d(
        X_train, X_test, y_train, y_test, pred_grid, (xx, yy), X_val
    )
    plt.savefig("example_class_full_2d.png")


if __name__ == "__main__":
    run_example_data()
