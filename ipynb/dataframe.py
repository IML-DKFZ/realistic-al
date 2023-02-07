from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, List, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from rich.pretty import pprint

import ipynb_setup

from plotlib.performance_plots import plot_standard_dev
from utils.file_utils import get_experiment_df


def get_exp_paths(base_path: Path, dataset_dict: Dict[str, str]):
    exp_paths = []
    for datapath in base_path.iterdir():
        if datapath.name in dataset_dict.values() and datapath.is_dir():
            for regime_path in datapath.iterdir():
                # if regime_path.name != "full_data" and regime_path.is_dir():
                if "active" in regime_path.name and regime_path.is_dir():
                    for exp_path in regime_path.iterdir():
                        if exp_path.is_dir():
                            exp_paths.append(exp_path)
    return exp_paths


def get_exp_df(exp_path: Path):
    experiment_frame = get_experiment_df(exp_path, name=None)
    if experiment_frame is None:
        return None

    # Add new metric values
    experiment_add = get_experiment_df(exp_path, pattern="test_metrics.csv", name=None)
    if experiment_add is not None:
        del experiment_add["Name"]
        del experiment_add["version"]
        experiment_frame = experiment_frame.join(experiment_add)

    experiment_frame["Path"] = str(exp_path)

    return experiment_frame


def create_experiment_df(
    base_path: Path, dataset_dict: Dict[str, str], rewrite: bool = True,
):

    save_file = Path("./plots/save.pkl")
    # if save_file.exists() and rewrite is False:
    #     df = pd.read_pickle(save_file)
    #     try:
    #         df

    if not save_file.exists() or rewrite:
        exp_paths = get_exp_paths(base_path, dataset_dict)

        df = []
        for exp_path in exp_paths:
            exp_df = get_exp_df(exp_path)
            df.append(exp_df)
        df = pd.concat(df)
        df.reset_index(inplace=True)
        df.to_pickle(save_file)
    else:
        df = pd.read_pickle(save_file)

    # errors can occur here if base_path is incorrect for loaded experiments

    df["Rel. Path"] = df["Path"].map(lambda x: x.split(str(base_path) + "/")[1])

    return df


def preprocess_df(
    df: pd.DataFrame,
    match_patterns: List[str],
    df_funcdict: Dict[str, Callable] = {
        "Training": lambda x: "PT: {}, Sem-SL: {}".format(x["Self-SL"], x["Semi-SL"])
    },
):
    df["Dataset"] = df["Rel. Path"].map(lambda x: x.split("/")[0])
    df["Label Regime"] = df["Rel. Path"].map(lambda x: x.split("/")[1])
    df["Experiment Name"] = df["Rel. Path"].map(lambda x: x.split("/")[2])

    matches = get_experiments_matching(df, "Experiment Name", match_patterns)
    df_match = df[matches]

    # Leads to no sliced dataframe errors:
    # See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    #  value is trying to be set on a copy of a slice from a DataFrame.
    # Try using .loc[row_indexer,col_indexer] = value instead
    df_match = df_match.copy(deep=True)

    df_match["Query Method"] = df_match["Experiment Name"].apply(
        lambda x: x.split("acq-")[1].split("_")[0]
    )
    df_match["Self-SL"] = df_match["Experiment Name"].apply(
        lambda x: "pretrained_model" in x
    )
    df_match["Semi-SL"] = df_match["Experiment Name"].apply(lambda x: "fixmatch" in x)

    for key, function in df_funcdict.items():
        df_match[key] = df_match.apply(function, axis=1)
    # df_match[style_name] = df_match.apply(
    #     lambda x: "PT: {}, Sem-SL: {}".format(x["Self-SL"], x["Semi-SL"]), axis=1
    # )

    return df_match


def create_plot_df(
    df: pd.DataFrame,
    settings_dict: Dict[str, List[str]],
    filter_dict: Dict[str, List[str]],
):
    """Create the Dataframe for a single plot

    Args:
        df (pd.DataFrame): Input Dataframe
        settings_dict (Dict[str, List[str]]): Dictionary with for matching settings
        filter_dict (str, List[str]]): Dictionary for regex fitlering values

    Returns:
        _type_: _description_

    -------
    e.g.
    filter_dict = {
        "Rel. Path" : [".*batchbald.*"]
    }
    settings_dict = {
        "Dataset": ["cifar10"],
        "Label Regime": ["Low"],
    }

    """
    df_out = df.copy(deep=True)
    # Filter out all experiments matching filter patterns in filter_dict

    for filter_key, filter_patterns in filter_dict.items():
        # Get inverted values
        filter_matches = get_experiments_matching(df, filter_key, filter_patterns).map(
            lambda x: x is False
        )
        df_out = df_out[filter_matches]

    # Only take experiments matching settings values
    for key, val in settings_dict.items():
        df_out = df_out[df_out[key].isin(val)]

    return df_out


def get_experiments_matching(df: pd.DataFrame, key: str, patterns: List[str]):
    """Returns true for each row in key if any of the patterns are True.
    If patterns is empty, returns True for all.

    Args:
        df (pd.DataFrame): _description_
        key (str): _description_
        patterns (List[str]): _description_

    Returns:
        pd.Series: can be used df[matches] to keep rues
    """
    # Returns true if any of the patterns are true using piping |
    # Source: https://stackoverflow.com/questions/8888567/match-a-line-with-multiple-regex-using-python/8888615#8888615
    matches = df[key].str.match("|".join(patterns))
    return matches


def main():
    parser = ArgumentParser()
    # parser.add_argument("-d", "--dataset", type=str, choices=list(DATASETS.keys()))
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

    # print(args)

    # df = create_experiment_df(Path(args.base_path))
    # df = preprocess_df(df, MATCH_PATTERNS)
    # settings_dict = {"Dataset": ["cifar10"], "Label Regime": ["active-cifar10_low"]}
    # filter_dict = {"Rel. Path": [r".*batchbald.*"]}
    # style_name = "Training"
    # plot_df = create_plot_df(df, settings_dict, filter_dict)

    # fig, ax = create_plots(plot_df, style_name)
    # plt.savefig("./plots/test.pdf")


# if __name__ == "__main__":
#     main()

