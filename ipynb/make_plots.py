from argparse import ArgumentParser
import re
from itertools import product
from pathlib import Path
from typing import Callable, List, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.pretty import pprint

import ipynb_setup

from plotlib.performance_plots import plot_standard_dev
from utils.file_utils import get_experiment_df

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
        "Basic": "basic_model-resnet_drop-0_aug-cifar_randaugmentMC_wd-0.0005_lr-0.1_optim-sgd_cosine_weighted-true",
        "PT": None,
    },
    "MIO-TCD": {
        "Basic": "basic_model-resnet_drop-0_aug-imagenet_randaugMC_wd-5e-05_lr-0.1_optim-sgd_cosine_weighted-True_epochs-80",
        "PT": None,
    },
    "ISIC-2019": {
        "Basic": "basic_model-resnet_drop-0_aug-isic_train_wd-0.005_lr-0.01_optim-sgd_cosine_weighted-True_epochs-200",
        "PT": None,
    },
}
PALETTE = {
    "bald": "tab:blue",
    "kcentergreedy": "tab:green",
    "entropy": "tab:orange",
    "random": "tab:red",
    "batchbald": "tab:cyan",
}

DASHES = {
    "PT: False, Sem-SL: False": (4, 4),
    "PT: True, Sem-SL: False": (1, 0),
    "PT: False, Sem-SL: True": (1, 2),
    "PT: True, Sem-SL: True": (2, 1),
}

MARKERS = {
    "PT: False, Sem-SL: False": "v",
    "PT: True, Sem-SL: False": "o",
    "PT: False, Sem-SL: True": "D",
    "PT: True, Sem-SL: True": "X",
}

TRAINING_SETTINGS = {
    "all": [
        "PT: False, Sem-SL: False",
        "PT: True, Sem-SL: False",
        "PT: False, Sem-SL: True",
        "PT: True, Sem-SL: True",
    ],
    "basic": ["PT: False, Sem-SL: False"],
    "Self-SL": ["PT: True, Sem-SL: False"],
    "Sem-SL": ["PT: False, Sem-SL: True"],
    "Self-Sem-SL": ["PT: True, Sem-SL: True"],
}

PLOT_VALUES = {
    "Accuracy": "test_acc",
    "Balanced Accuracy": "test/w_acc",
    "Batch Entropy": "Acquisition Entropy",
    "Acquired Entropy": "Dataset Entropy",
}

MATCH_PATTERNS = [
    r"basic_.*",
    r"basic-pretrained_.*",
    #     r".*__wloss.*"
    #     BB experiment
    #     r".*bald.*"
    #     r".*random.*"
    r"fixmatch_.*",
    #     r"fixmatch-pretrained_.*",
]

FILTER_DICT = {"standard": [".*batchbald.*"], "bb": []}

UNIT_VALS = None
DATASETS = {
    "CIFAR-10": "cifar10",
    "CIFAR-100": "cifar100",
    "CIFAR-10-LT": "cifar10_imb",
    "MIO-TCD": "miotcd",
    "ISIC-2019": "isic2019",
}


def path_to_style_str(path: Path) -> str:
    return f"PT: {'pretrained_model' in path.name}, Sem-SL: {'fixmatch' in path.name}"


def dataset_name_to_id(dataset_name: str) -> Tuple[str, str]:
    """Convert human level dataset name to computer-readable dataset identifier.
    Also, select fill set and check if it is a subset.

    Args:
        dataset_name: human-level name of the dataset

    Returns:
        normalized dataset and fillset identifiers
    """

    sub_datasets = {"CIFAR-10-LT": DATASETS["CIFAR-10"], "ISIC-2019": "isic19"}

    dataset_id = DATASETS[dataset_name]
    fillset_id = sub_datasets.get(dataset_name, dataset_id)

    return dataset_id, fillset_id


def build_setting_path_mapping(
    dataset_id: str, fillset_id: str, base_path: Path
) -> Dict[str, List[Path]]:
    """Create a dict that maps experiment settings to data paths.

    Args:
        dataset_id: normalized dataset name for this group of experiments
        fillset_id: normalized fillset name for this group of experiments
        base_path: path where all experiments are saved

    Returns:
        a mapping from experiment settings to folders of experiment data
    """
    settings = {
        "full": ["low", "med", "high"],
        "low-bb": ["low-batchbald"],
        "low-bb-fulll": ["low-batchbald", "low"],
        "bblow": ["low"],
    }

    for setting in settings:
        settings[setting] = [f"active-{fillset_id}_{exp}" for exp in settings[setting]]

    setting_paths: Dict[str, List[Path]] = {}

    for setting, experiments in settings.items():
        setting_paths[setting] = [base_path / dataset_id / exp for exp in experiments]

    return setting_paths


def load_experiment_from_path(
    base_path: Path,
    filter_patterns: List[str],
    hue_name: str,
    hue_split: str,
    style_fct: Callable[[Path], str],
    style_name: str,
    unit_name: str,
):
    """Returns a joint dataset for each experiment row e.g. cifar10_low

    Args:
        base_path (Path): path to experiment row
        filter_patterns (List[str]): pattern according to which naming is done
        hue_name (str): name of hue_name in dataframe
        hue_split (str): spliting of path name for hue_name creation
        style_fct (Callable[[Path], str]): function creating styles in dataframe
        style_name (str): name of style in dataframe
        unit_name (str): name of unit_val in dataframe for singular plots

    Returns:
        DataFrame used for plotting of an experiment row
    """
    paths = [path for path in base_path.iterdir() if path.is_dir()]
    paths.sort()
    print(f"Folders in Path: \n {base_path}\n")

    experiment_paths: List[Path] = []

    experiment_paths = list(
        filter(
            lambda path: any(
                re.match(pattern, str(path.name)) is not None
                for pattern in MATCH_PATTERNS
            ),
            paths,
        )
    )
    experiment_paths = list(
        filter(
            lambda path: all(
                re.match(pattern, str(path.name)) is None for pattern in filter_patterns
            ),
            experiment_paths,
        )
    )

    hue_names = [
        path.name.split(hue_split)[1].split("_")[0] for path in experiment_paths
    ]
    style_vals = [style_fct(path) for path in experiment_paths]

    df = []
    for i, (exp_path) in enumerate(experiment_paths):
        exp_path = Path(exp_path)
        hue_val = hue_names[i]
        style_val = style_vals[i]
        if UNIT_VALS is not None:
            unit_val = UNIT_VALS[i]
        else:
            unit_val = None

        experiment_frame = get_experiment_df(exp_path, name=hue_val)
        if experiment_frame is None:
            continue

        # Add new metric values
        experiment_add = get_experiment_df(
            exp_path, pattern="test_metrics.csv", name=hue_val
        )
        if experiment_add is not None:
            del experiment_add["Name"]
            del experiment_add["version"]
            experiment_frame = experiment_frame.join(experiment_add)

        experiment_frame[hue_name] = hue_val
        experiment_frame[style_name] = style_val
        experiment_frame[unit_name] = unit_val

        df.append(experiment_frame)

    df = pd.concat(df)
    df.reset_index(inplace=True)

    return df


def get_key_filter(setting: str):
    key_filter = "standard"
    for key in FILTER_DICT:
        if key in setting:
            key_filter = key
    return key_filter


def create_plots_from_settings(
    dataset: str,
    full_data_dict,
    hue_name,
    savepath,
    setting_dfs,
    style_name,
    unit_name,
):
    upper_bounds = [True, False]
    y_shareds = [True, False]

    for y_shared, upper_bound in product(upper_bounds, y_shareds):
        for plot_val, plot_key in PLOT_VALUES.items():
            if plot_val in ["Batch Entropy", "Acquired Entropy"]:
                if not (y_shared and upper_bound):
                    continue

            for setting in setting_dfs:
                dfs = setting_dfs[setting]
                if plot_key in dfs[0]:
                    for training_setting, training_styles in TRAINING_SETTINGS.items():
                        num_cols = len(dfs)
                        ax_legend = 0
                        fig, axs = plt.subplots(ncols=num_cols, sharey=y_shared)
                        if num_cols == 1:
                            axs = [axs]
                        if UNIT_VALS is None:
                            unit_name = None

                        for i in range(num_cols):
                            df = dfs[i][dfs[i][style_name].isin(training_styles)]
                            if len(df) == 0:
                                continue
                            ax = axs[i]

                            legend = False
                            if i == ax_legend:
                                legend = "auto"
                            ax = plot_standard_dev(
                                ax,
                                df,
                                y=plot_key,
                                hue=hue_name,
                                style=style_name,
                                units=unit_name,
                                ci="sd",
                                legend=legend,
                                palette=PALETTE,
                                markers=MARKERS,
                                dashes=DASHES,
                                err_kws={"alpha": 0.2},
                            )  # , units=unit_name)
                            full_dict = {
                                "Basic": {"color": "black", "linestyle": "--"},
                                "PT": {"color": "black", "linestyle": "- "},
                            }
                            if upper_bound:
                                for model in full_dict:
                                    if model in full_data_dict:
                                        if plot_key in full_data_dict[model]:
                                            ax.axhline(
                                                full_data_dict[model][plot_key]["mean"]
                                                / 100,
                                                **full_dict[model],
                                            )
                            ax.set_xlabel("Labeled Samples")
                            ax.set_ylabel(plot_val)
                            if i == ax_legend:
                                ax.get_legend().remove()
                        if num_cols == 3:
                            ncol_legend = 5
                        else:
                            ncol_legend = 3
                        fig.legend(
                            loc="upper center",
                            bbox_to_anchor=(0.5, -0.05),
                            fancybox=True,
                            shadow=True,
                            ncol=ncol_legend,
                        )

                        fig.set_size_inches(4 * num_cols, 6)
                        fig.tight_layout()
                        save_dir = savepath / dataset / plot_val.replace(" ", "-")
                        if not save_dir.is_dir():
                            save_dir.mkdir(parents=True)
                        if upper_bound is True and y_shared is True:
                            fn = f"plot-{setting}_train-{training_setting}.pdf"
                        else:
                            fn = f"plot-{setting}_train-{training_setting}_yshared-{y_shared}_bound-{upper_bound}.pdf"
                        print(f"Filename:{fn}")
                        plt.savefig(save_dir / fn, bbox_inches="tight")


def load_full_data(base_path: Path, d_set: str, dataset: str):
    """Load plots of experiments ran on the whole dataset.

    Args:
        base_path (Path): path to the experiment folder
        d_set (str): dataset name machine readable
        dataset (str): dataset name

    Returns:
        dictionary of metrics on the whole dataset
    """
    full_paths = {}
    for model in FULL_MODELS[dataset]:
        if FULL_MODELS[dataset][model] is not None:
            full_paths[model] = (
                base_path / d_set / "full_data" / FULL_MODELS[dataset][model]
            )

    print(full_paths)
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


def load_settings_data(
    hue_name: str,
    hue_split: str,
    setting_paths: Dict[str, List[Path]],
    style_fct: Callable[[Path], str],
    style_name: str,
    unit_name: str,
) -> Dict[str, List[pd.DataFrame]]:
    """Creates a list of 

    Args:
        hue_name (str): _description_
        hue_split (str): _description_
        setting_paths (Dict[str, List[Path]]): _description_
        style_fct (Callable[[Path], str]): _description_
        style_name (str): _description_
        unit_name (str): _description_

    Returns:
        Dict[str, List[pd.DataFrame]]: _description_
    """
    setting_dfs: Dict[str, List[pd.DataFrame]] = {}

    for setting, base_paths in setting_paths.items():

        if not all(base_path.is_dir() for base_path in base_paths):
            print(f"Skipping Setting: {setting}\nPath is not existent {base_path}")
            continue

        key_filter = get_key_filter(setting)
        print(f"Selecting Filter Pattern from {key_filter}")
        filter_patterns = FILTER_DICT[key_filter]

        setting_dfs[setting] = []

        for base_path in base_paths:
            dataframe = load_experiment_from_path(
                base_path,
                filter_patterns,
                hue_name,
                hue_split,
                style_fct,
                style_name,
                unit_name,
            )

            setting_dfs[setting].append(dataframe)
    return setting_dfs


def make_plots_for_dataset(
    dataset: str,
    base_path: Path,
    save_path: Path,
    hue_name: str,
    hue_split: str,
    style_name: str,
    unit_name: str,
):
    sns.set_style("whitegrid")

    d_set, fill_set = dataset_name_to_id(dataset)
    setting_paths = build_setting_path_mapping(d_set, fill_set, base_path)
    pprint(setting_paths)

    setting_dfs = load_settings_data(
        hue_name, hue_split, setting_paths, path_to_style_str, style_name, unit_name,
    )

    full_data_dict = load_full_data(base_path, d_set, dataset)

    create_plots_from_settings(
        dataset,
        full_data_dict,
        hue_name,
        save_path,
        setting_dfs,
        style_name,
        unit_name,
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, choices=list(DATASETS.keys()))
    parser.add_argument(
        "-s", "--save-path", type=Path, default=Path("./plots").resolve()
    )
    parser.add_argument(
        "-b",
        "--base-path",
        type=Path,
        default=Path(
            "~/network/Personal/carsten_al_cvpr_2023-November/logs_cluster/activelearning"
        ).expanduser(),
    )
    parser.add_argument("--hue-name", type=str, default="Acquisition")
    parser.add_argument("--hue-split", type=str, default="acq-")
    parser.add_argument(
        "--style-name", type=str, default="PreTraining & Semi-Supervised"
    )
    parser.add_argument("--unit-name", type=str, default="Unit")
    args = parser.parse_args()

    make_plots_for_dataset(
        dataset=args.dataset,
        base_path=args.base_path,
        save_path=args.save_path,
        hue_name=args.hue_name,
        hue_split=args.hue_split,
        style_name=args.style_name,
        unit_name=args.unit_name,
    )


if __name__ == "__main__":
    main()
