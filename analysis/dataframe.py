from pathlib import Path
from typing import Callable, Dict, List

# required to access files in src
import ipynb_setup
import pandas as pd

from utils.file_utils import get_experiment_df


def get_exp_paths(base_path: Path, dataset_dict: Dict[str, str]) -> List[Path]:
    """
    Retrieves a list of experiment paths based on a base path and a dataset dictionary.

    Args:
        base_path (Path): The base directory path where experiments are stored.
        dataset_dict (Dict[str, str]): A dictionary mapping dataset names to their corresponding paths.

    Returns:
        List[Path]: A list of Path objects representing experiment paths.

    Example:
        >>> base_path = Path('/path/to/experiments')
        >>> dataset_dict = {'dataset1': 'path/to/dataset1', 'dataset2': 'path/to/dataset2'}
        >>> exp_paths = get_exp_paths(base_path, dataset_dict)
        >>> exp_paths
        [Path('/path/to/experiments/dataset1/active/experiment1'),
         Path('/path/to/experiments/dataset2/active/experiment2')]
    """
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


def get_exp_df(exp_path: Path) -> pd.DataFrame:
    """
    Retrieves a pandas DataFrame representing experiment data from the given experiment path.
    Patterns which are searched for below the path: `stored.npz` and `test_metrics.csv`

    Args:
        exp_path (Path): The path to the experiment directory.

    Returns:
        pd.DataFrame or None: A DataFrame containing experiment data, or None if no data is found.

    Example:
        >>> exp_path = Path('/path/to/experiment')
        >>> experiment_data = get_exp_df(exp_path)
        >>> print(experiment_data)
           Metric1  Metric2  Metric3            Path
        0      0.1      0.5      0.3  /path/to/experiment
    """
    experiment_frame = get_experiment_df(exp_path, pattern="stored.npz", name=None)
    if experiment_frame is None:
        return None

    # Add new metric values
    experiment_add = get_experiment_df(exp_path, pattern="test_metrics.csv", name=None)
    if experiment_add is not None:
        del experiment_add["Name"]
        del experiment_add["version"]
        # do not use join here -- can sometimes lead to problems
        for col in experiment_add.columns:
            experiment_frame[col] = experiment_add[col]

    experiment_frame["Path"] = str(exp_path)

    return experiment_frame


def create_experiment_df(
    base_path: Path,
    dataset_dict: Dict[str, str],
    rewrite: bool = True,
    save_file: str = "./plots/safe_df.pkl",
) -> pd.DataFrame:
    """
    Creates or loads a pandas DataFrame representing experiment data from the given base path and dataset dictionary.

    Args:
        base_path (Path): The base directory path where experiments are stored.
        dataset_dict (Dict[str, str]): A dictionary mapping dataset names to their corresponding paths.
        rewrite (bool, optional): If True, recreate the DataFrame even if a saved file exists. Default is True.
        save_file (str, optional): The file path to save or load the DataFrame. Default is "./plots/safe_df.pkl".

    Returns:
        pd.DataFrame: A DataFrame containing experiment data.

    Raises:
        FileNotFoundError: If no saved DataFrame file is found and `rewrite` is set to False.

    Example:
        >>> base_path = Path('/path/to/experiments')
        >>> dataset_dict = {'dataset1': 'path/to/dataset1', 'dataset2': 'path/to/dataset2'}
        >>> experiment_data = create_experiment_df(base_path, dataset_dict)
        >>> print(experiment_data)
           index  Metric1  Metric2  Metric3            Path  Rel. Path
        0      0      0.1      0.5      0.3  /path/to/experiment  dataset1/active/experiment1
        1      0      0.2      0.6      0.4  /path/to/experiment  dataset2/active/experiment2
    """
    save_file = Path(save_file)

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
) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by adding columns for dataset, label regime, and experiment name,
    and performs additional transformations based on matching patterns.

    Args:
        df (pd.DataFrame): The input DataFrame containing experiment data.
        match_patterns (List[str]): List of patterns to filter and match experiment names.
        df_funcdict (Dict[str, Callable], optional): A dictionary of functions to apply to the DataFrame.
            Default is {"Training": lambda x: "PT: {}, Sem-SL: {}".format(x["Self-SL"], x["Semi-SL"])}.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    Example:
        >>> df = create_experiment_df(base_path, dataset_dict)
        >>> match_patterns = ['acq-rand', 'acq-entropy']
        >>> preprocessed_df = preprocess_df(df, match_patterns)
        >>> print(preprocessed_df)
           index  Metric1  Metric2  Metric3  ...  Semi-SL           Training
        0      0      0.1      0.5      0.3  ...   False  PT: False, Sem-SL: True
        1      0      0.2      0.6      0.4  ...    True  PT: True, Sem-SL: True
    """
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

    return df_match


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


def create_plot_df(
    df: pd.DataFrame,
    settings_dict: Dict[str, List[str]],
    filter_dict: Dict[str, List[str]],
):
    """
    Creates a filtered DataFrame for generating plots based on settings and filter criteria.

    Args:
        df (pd.DataFrame): The input DataFrame containing experiment data.
        settings_dict (Dict[str, List[str]]): A dictionary of settings to include in the filtered DataFrame.
        filter_dict (Dict[str, List[str]]): A dictionary of filter criteria to exclude from the DataFrame.

    Returns:
        pd.DataFrame: The filtered DataFrame for generating plots.

    Example:
        >>> df = preprocess_df(df, match_patterns)
        >>> settings_dict = {'Query Method': ['rand', 'entropy'], 'Training': ['PT: False, Sem-SL: True']}
        >>> filter_dict = {'Label Regime': ['active']}
        >>> plot_df = create_plot_df(df, settings_dict, filter_dict)
        >>> print(plot_df)
           index  Metric1  Metric2  Metric3  ...  Semi-SL           Training  Dataset        ...
        0      0      0.1      0.5      0.3  ...   False  PT: False, Sem-SL: True  dataset1   ...
        1      0      0.2      0.6      0.4  ...    True  PT: False, Sem-SL: True  dataset1   ...
        ...
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


def get_experiments_matching(
    df: pd.DataFrame, key: str, patterns: List[str]
) -> pd.Series:
    """
    Returns a boolean Series indicating whether each row in the specified column matches any of the given patterns.

    If patterns is empty, returns True for all rows.

    Args:
        df (pd.DataFrame): The input DataFrame containing experiment data.
        key (str): The column name to match against.
        patterns (List[str]): A list of patterns to match.

    Returns:
        pd.Series: A boolean Series, where True indicates a match.

    Example:
        >>> df = preprocess_df(df, match_patterns)
        >>> patterns = ['rand', 'entropy']
        >>> matches = get_experiments_matching(df, 'Query Method', patterns)
        >>> filtered_df = df[matches]
        >>> print(filtered_df)
           index  Metric1  Metric2  Metric3  ...  Semi-SL           Training  Dataset        ...
        0      0      0.1      0.5      0.3  ...   False  PT: False, Sem-SL: True  dataset1   ...
        1      0      0.2      0.6      0.4  ...    True  PT: False, Sem-SL: True  dataset1   ...
        ...
    """
    # Returns true if any of the patterns are true using piping |
    # Source: https://stackoverflow.com/questions/8888567/match-a-line-with-multiple-regex-using-python/8888615#8888615
    matches = df[key].str.match("|".join(patterns))
    return matches
