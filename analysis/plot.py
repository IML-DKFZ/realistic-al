import os
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Union

# required for accessing functions from src
import ipynb_setup
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dataframe import *
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from plotlib.performance_plots import plot_standard_dev
from utils.eval import get_aubc

#### Set Style
sns.set_style("whitegrid")
plt.rc("font", size=12)

### Set Paths
# contains RESULTSPATH/DATASET/LABEL-REGIME/EXPERIMENT/SEEDS/LOOPS
RESULTSPATH = "/mnt/drive_nvme2/logs_cluster/activelearning"

# will be filled with results
SAVEPATH = "./plots"

### Paths to full modles.
FULL_MODELS = {
    "CIFAR-10": {
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.0005_lr-0.1_optim-sgd_cosine",
        "PT": None,
    },
    "CIFAR-100": {
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.005_lr-0.1_optim-sgd_cosine",
        "PT": None,
    },
    "CIFAR-10-LT": {
        # WLoss
        # "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.0005_lr-0.1_optim-sgd_cosine_weighted-true",
        # "PT": None,
        # Balanced
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.005_lr-0.1_optim-sgd_cosine_balancsamp-True",
        "PT": None,
    },
    "MIO-TCD": {
        # WLoss
        # "Basic": "basic_model-resnet_drop-0_aug-imagenet_randaugMC_wd-5e-05_lr-0.1_optim-sgd_cosine_weighted-True_epochs-80",
        # "PT": None,
        # Balanced
        "Basic": "basic_model-resnet_drop-0_aug-imagenet_randaugMC_wd-0.0005_lr-0.1_optim-sgd_cosine_weighted-False_epochs-80_balancsamp-True",
        "PT": None,
    },
    "ISIC-2019": {
        # WLoss
        # "Basic": "basic_model-resnet_drop-0_aug-isic_train_wd-0.005_lr-0.01_optim-sgd_cosine_weighted-True_epochs-200",
        # "PT": None,
        # Balanced
        "Basic": "basic_model-resnet_drop-0_aug-isic_randaugmentMC_wd-0.0005_lr-0.1_optim-sgd_cosine_weighted-False_epochs-200_balancsamp-True",
        "PT": None,
    },
}

QUERYMETHODS = {
    "bald": "BALD",
    "kcentergreedy": "Core-Set",
    "entropy": "Entropy",
    "random": "Random",
    "badge": "BADGE",
    "batchbald": "BatchBALD",
}

PALETTE = {
    "BALD": "tab:blue",
    "Core-Set": "tab:green",
    "Entropy": "tab:orange",
    "BADGE": "tab:purple",
    "Random": "tab:red",
    "BatchBALD": "tab:cyan",
}


def style_func_dict(x: Dict[str, bool]) -> str:
    """Function used on dictionaries and dataframes to generate the style value for a plot."""
    out = []
    if x["Self-SL"]:
        out.append("Self-SL Pre-Trained")
    if x["Semi-SL"]:
        out.append("Semi-SL")
    if len(out) == 0:
        out.append("Standard Training")

    return " ".join(out)


# dash styles in final plot
DASHES = {
    style_func_dict({"Self-SL": False, "Semi-SL": False}): (4, 4),
    style_func_dict({"Self-SL": True, "Semi-SL": False}): (1, 0),
    style_func_dict({"Self-SL": False, "Semi-SL": True}): (1, 2),
    style_func_dict({"Self-SL": True, "Semi-SL": True}): (2, 1),
}

# marker styles in final plot
MARKERS = {
    style_func_dict({"Self-SL": False, "Semi-SL": False}): "v",
    style_func_dict({"Self-SL": True, "Semi-SL": False}): "o",
    style_func_dict({"Self-SL": False, "Semi-SL": True}): "D",
    style_func_dict({"Self-SL": True, "Semi-SL": True}): "X",
}

# which dataset to use
DATASETS = {
    "CIFAR-10": "cifar10",
    "CIFAR-100": "cifar100",
    "CIFAR-10-LT": "cifar10_imb",
    "MIO-TCD": "miotcd",
    "ISIC-2019": "isic2019",
}

# required parameters
MATCH_PATTERNS = [
    r"basic_.*",
    r"basic-pretrained_.*",
    r"fixmatch_.*",
]


# given a setting filter out values with
FILTERDICT = {
    "full": {"Rel. Path": [".*batchbald.*"]},
    "bblow": {},
    "bb": {},
    "abl_cifar100": {},
    "abl_isic": {},
}

# Style based on this key
STYLENAME = "Training"

# Overwrite values
VALUE_DICT = {STYLENAME: style_func_dict}


def get_label_regime(x: str) -> str:
    """Function obtaining the label regime names.
    eg. active-cifar10_low to low

    Args:
        x (str): _description_

    Returns:
        str: _description_
    """
    out = "_".join([x.split("_")[v] for v in range(1, len(x.split("_")))])
    if len(out) == 0:
        out = x
    return out


def load_full_data(
    base_path: Path, dataset: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Obtain performance values for models trained on the entire dataset.

    Args:
        base_path (Path): _description_
        dataset (str): _description_

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: nested dictionary with performance metrics.
    """
    full_paths = {}
    for model in FULL_MODELS[dataset]:
        if FULL_MODELS[dataset][model] is not None:
            full_paths[model] = (
                base_path
                / DATASETS[dataset]
                / "full_data"
                / FULL_MODELS[dataset][model]
            )

    full_data_dict = {}

    for key, path in full_paths.items():
        test_acc_df = pd.read_csv(path / "test_acc.csv", index_col=0)
        full_data_dict[key] = {}
        full_data_dict[key]["test_acc"] = {}
        full_data_dict[key]["test_acc"]["mean"] = float(test_acc_df["Mean"])
        full_data_dict[key]["test_acc"]["std"] = float(test_acc_df["STD"])

        if (path / "test_w_acc.csv").is_file():
            mean_recall_df = pd.read_csv(path / "test_w_acc.csv", index_col=0)
            full_data_dict[key]["test/w_acc"] = {}
            full_data_dict[key]["test/w_acc"]["mean"] = float(mean_recall_df["Mean"])
            full_data_dict[key]["test/w_acc"]["std"] = float(mean_recall_df["STD"])
    return full_data_dict


def plot_experiment(
    dfs: List[pd.DataFrame],
    hue_name: str,
    style_name: str,
    plot_key: str,
    upper_bound: bool = False,
    sharey=True,
    num_cols: int = None,
    full_models_dict: Dict[str, float] = None,
    ylabel: str = None,
    xlabel: str = None,
    ax_legend: int = 0,
) -> Tuple[Figure, List[Axes]]:
    """Plotting Script for whole Experiment rows.
    Resulting in a figure with multiple plots.
    If some dfs are missing to fill num_cols then they are filled up
    and branded.

    Args:
        dfs (List[pd.DataFrame]): Dataframes for plot
        hue_name (str): Color key
        style_name (str): Style key
        plot_key (str): y-axis key
        upper_bound (bool, optional): Add performance of full model. Defaults to False.
        sharey (bool, optional): plots share y range. Defaults to True.
        num_cols (Optional[int], optional): Number of Columns. Defaults to None.
        full_models_dict (Dict[str, float], optional): Dictionary containing values for full models. Defaults to None.
        ylabel (str, optional): Defaults to None.
        xlabel (str, optional): Defaults to None.
        ax_legend (int, optional): from which ax the legend is taken. Defaults to 0.

    Returns:
        Tuple[Figure, List[Axes]]: Figures and corresponding plots
    """
    if num_cols is None:
        num_cols = len(dfs)
    fig, axs = plt.subplots(ncols=num_cols, sharey=sharey, figsize=(4 * num_cols, 6))
    if num_cols == 1:
        axs = [axs]

    # Get all settings and values for axhline
    full_dict = {
        "Basic": {"color": "black", "linestyle": "--"},
        "PT": {"color": "black", "linestyle": "- "},
    }
    print(full_models_dict)
    hline_dicts = []
    for model in full_dict:
        if model in full_models_dict:
            if plot_key in full_models_dict[model]:
                hline_dict = dict(
                    y=full_models_dict[model][plot_key]["mean"] / 100,
                    **full_dict[model],
                )
                hline_dicts.append(hline_dict)

    for i in range(num_cols):
        if i < len(dfs):
            df = dfs[i]
        else:
            axs[i].text(0.2, 0.5, "No Experiments Performed")
            axs[i].set_xticks([])
            continue

        legend = False
        if i == ax_legend:
            legend = "auto"
        plot_standard_dev(
            axs[i],
            df,
            y=plot_key,
            hue=hue_name,
            style=style_name,
            # units=unit_name,
            ci="sd",
            legend=legend,
            palette=PALETTE,
            markers=MARKERS,
            dashes=DASHES,
            err_kws={"alpha": 0.2},
        )  # , units=unit_name)

        # fat borders
        [x.set_linewidth(2) for x in axs[i].spines.values()]

    if sharey is True:
        fig.subplots_adjust(wspace=0.075, hspace=0)
    else:
        fig.tight_layout()
    if upper_bound:
        # Plot axhline into all values
        add_upper_bound(axs, hline_dicts)

    set_legend(ylabel, xlabel, fig, axs)

    return fig, axs


def set_legend(ylabel: str, xlabel: str, fig: Figure, axs: List[Axes]) -> None:
    """Sets the labels and removes legends on axes and sets legend below figure

    ___
    Note: it is best to have only one legend on all axs if identical

    Args:
        ylabel (str): ylabel name
        xlabel (str): xlabel name
        fig (Figure): Figure to be changed
        axs (List[Axes]): axes belonging to figure
    """
    num_cols = len(axs)
    for i in range(num_cols):
        if xlabel is not None:
            axs[i].set_xlabel(xlabel)
        if ylabel is not None:
            axs[i].set_ylabel(ylabel)
        if axs[i].get_legend() is not None:
            axs[i].get_legend().remove()
    if num_cols == 3:
        ncol_legend = 5
    else:
        ncol_legend = 3
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        fancybox=True,
        shadow=True,
        ncol=ncol_legend,
    )


def add_upper_bound(axs: List[plt.Axes], hline_dicts: List[Dict[str, float]]):
    """Add upper bound to axes with axhline given a list of dictionary with alles inputs for ax.hline

    Args:
        axs (List[plt.Axes]): axes to be annotated
        hline_dicts (List[Dict[any]]): must contain {"y" : flaot} and can contain other values
    """
    num_cols = len(axs)
    for i in range(num_cols):
        for hline_dict in hline_dicts:
            axs[i].axhline(**hline_dict)


def to_path_name(string: str) -> str:
    """Converts a given string to a valid path name by replacing spaces with hyphens,
    converting to lowercase, and removing periods.

    Args:
        string (str): The input string to be converted.

    Returns:
        str: The resulting path name.

    Example:
        >>> to_path_name("Hello World.txt")
        'hello-worldtxt'

    """
    out = string.replace(" ", "-")
    out = out.lower()
    out = out.replace(".", "")
    return out


def save_plot(
    save_path: Path,
    save_dict: Dict[str, any],
):
    """
    Turns all values in the save_dict into a string as a name and saves the active plot.

    Args:
        save_path (Path): The directory path where the plot will be saved.
        save_dict (Dict[str, Any]): A dictionary containing values to be included in the plot name.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified save_path directory does not exist.

    Example:
        >>> save_dict = {'title': 'My Plot', 'size': (10, 8), 'color': 'blue'}
        >>> save_plot(Path('plots'), save_dict)
        # Saves the plot with a name like 'title-my-plot_size-(10,8)_color-blue.png'
    """
    save_name = []
    for key in save_dict:
        key_str = to_path_name(key)
        val_str = to_path_name(f"{save_dict[key]}")
        save_name.append(f"{key_str}-{val_str}")
    save_name = "_".join(save_name)
    if not save_path.is_dir():
        save_path.mkdir(parents=True)
    plt.savefig(save_path / save_name, bbox_inches="tight", pad_inches=0.01, dpi=600)


def product_of_dictionary(dictionary: Dict[str, Union[list, tuple]]) -> List[dict]:
    """Returns a list of dictionaries representing the Cartesian product of values in the input dictionary.

    Args:
        dictionary (Dict[str, Union[list, tuple]]): A dictionary where each key corresponds to a parameter
            and the associated value is a list or tuple of possible values for that parameter.

    Returns:
        List[dict]: A list of dictionaries representing all combinations of parameter values.

    Example:
        >>> dictionary = {"sharey": [True, False], "upper_bound": [True, False]}
        >>> list_of_products = product_of_dictionary(dictionary)
        >>> list_of_products
        [{'sharey': True, 'upper_bound': True},
         {'sharey': True, 'upper_bound': False},
         {'sharey': False, 'upper_bound': True},
         {'sharey': False, 'upper_bound': False}]
    """
    product_values = [x for x in product(*dictionary.values())]
    list_of_products = [
        dict(zip(dictionary.keys(), values)) for values in product_values
    ]
    return list_of_products


if __name__ == "__main__":
    RESULTSPATH = Path(RESULTSPATH)
    SAVEPATH = Path(SAVEPATH)
    if not SAVEPATH.exists():
        os.makedirs(SAVEPATH)
    df = create_experiment_df(
        RESULTSPATH, DATASETS, rewrite=True, save_file=SAVEPATH / "safe_df.pkl"
    )

    # df2 = create_experiment_df(base_path2, DATASETS, rewrite=True)

    # df = pd.concat([df, df2], axis=0)

    df = preprocess_df(df, MATCH_PATTERNS, VALUE_DICT)

    inv_dataset = {v: k for k, v in DATASETS.items()}

    df["Dataset"] = df["Dataset"].map(inv_dataset)
    df["Label Regime"] = df["Label Regime"].map(get_label_regime)
    df["Query Method"] = df["Query Method"].map(QUERYMETHODS)

    print(df["Query Method"].unique())

    full_models_dict = {
        dataset: load_full_data(RESULTSPATH, dataset) for dataset in DATASETS
    }

    settings_dict = {"Dataset": "CIFAR-10", "Label Regime": ["low"]}
    dataset_settings = {"Dataset": ["CIFAR-10", "CIFAR-100"]}
    plot_label_settings = {
        "full": {"Label Regime": [["low"], ["med"], ["high"]]},
        "bblow": {"Label Regime": [["low", "low-batchbald"]]},
        "bb": {"Label Regime": [["low-batchbald"]]},
        "abl_cifar100": {"Label Regime": [["low_qs-50"], ["low"], ["low_qs-2000"]]},
        "abl_isic": {"Label Regime": [["low_qs-10"], ["low_qs-40"], ["low_qs-160"]]},
    }

    plot_value_settings_dict = {
        "all": {"Self-SL": [True, False], "Semi-SL": [True, False]},
        "self-sl": {"Self-SL": [True], "Semi-SL": [False]},
        "standard": {"Self-SL": [False], "Semi-SL": [False]},
        "sem-sl": {"Self-SL": [False], "Semi-SL": [True]},
    }

    plot_settings = {"sharey": [True, False], "upper_bound": [True, False]}

    plot_settings_list = product_of_dictionary(plot_settings)

    plot_metrics_dict = {
        "CIFAR-10": [{"plot_key": "test_acc", "ylabel": "Accuracy"}],
        "CIFAR-100": [{"plot_key": "test_acc", "ylabel": "Accuracy"}],
        "CIFAR-10-LT": [{"plot_key": "test_acc", "ylabel": "Accuracy"}],
        "MIO-TCD": [{"plot_key": "test/w_acc", "ylabel": "Balanced Accuracy"}],
        "ISIC-2019": [{"plot_key": "test/w_acc", "ylabel": "Balanced Accuracy"}],
    }

    aubc_list = []

    for dataset in DATASETS:
        # currently only works for
        for label_setting_name, label_settings in plot_label_settings.items():
            full_dicts_plot = full_models_dict[dataset]
            for value_setting_name, value_setting in plot_value_settings_dict.items():
                df_plots = []
                for label_setting in label_settings["Label Regime"]:
                    # Alternatively this could also be changed up to using label_settings!
                    settings_dict = {
                        "Dataset": [dataset],
                        "Label Regime": label_setting,
                    }
                    settings_dict.update(value_setting)
                    df_plot = create_plot_df(
                        df,
                        settings_dict=settings_dict,
                        filter_dict=FILTERDICT[label_setting_name],
                    )
                    if len(df_plot) > 0:
                        df_plots.append(df_plot)
                print("Label Setting:")
                print(label_settings["Label Regime"])
                print(len(label_settings["Label Regime"]))
                print("-" * 8)

                if len(df_plots) > 0:
                    for df_plot in df_plots:
                        sortname = "Rel. Path"

                        for experiment_name in df_plot[sortname].unique():
                            use_df = df_plot[df_plot[sortname] == experiment_name]

                            num_iters = use_df["index"].unique().max() + 1
                            print(num_iters)
                            for plot_metric in plot_metrics_dict[dataset]:
                                perf_val = use_df[plot_metric["plot_key"]].to_numpy()
                                perf_val = perf_val.reshape(-1, num_iters)

                                for iteration in range(len(perf_val)):
                                    aubc = get_aubc(perf_val[iteration])
                                    aubc_row = {
                                        sortname: experiment_name,
                                        plot_metric["ylabel"]: aubc,
                                    }
                                    aubc_list.append(aubc_row)
                    aubc_df = pd.DataFrame(aubc_list)
                    aubc_df["Experiment Name"] = aubc_df[sortname].apply(
                        lambda x: x.split("/")[-1]
                    )
                    aubc_df = preprocess_df(aubc_df, MATCH_PATTERNS, VALUE_DICT)

                    for plot_metric in plot_metrics_dict[dataset]:
                        for plot_setting in plot_settings_list:
                            print(f"Dataset : {dataset}")
                            print(f"label_setting : {label_settings}")
                            print(f"Num Plots: {len(df_plots)}")
                            num_cols = len(label_settings["Label Regime"])
                            fig, axs = plot_experiment(
                                df_plots,
                                hue_name="Query Method",
                                style_name=STYLENAME,
                                full_models_dict=full_dicts_plot,
                                xlabel="Labeled Samples",
                                num_cols=num_cols,
                                **plot_metric,
                                **plot_setting,
                            )
                            plot_save_path = (
                                SAVEPATH
                                / dataset
                                / label_setting_name
                                / value_setting_name
                                / to_path_name(plot_metric["ylabel"])
                            )
                            save_plot(plot_save_path, plot_setting)
                            fig.clear()
    aubc_df = pd.DataFrame(aubc_list)
    aubc_df["Experiment Name"] = aubc_df[sortname].apply(lambda x: x.split("/")[-1])
    aubc_df = aubc_df.drop_duplicates()
    aubc_df = preprocess_df(aubc_df, MATCH_PATTERNS, VALUE_DICT)
    aubc_df.to_csv(SAVEPATH / "aubc.csv")
