"""
    Script to compute summary statistics for a given metric across multiple experiments.

    Usage:
        python script_name.py -p /path/to/experiments -l 1 -v test/acc -s auto

    Arguments:
        -p, --path: Base path containing experiment subdirectories.
        -l, --level: Number of folders to sweep. E.g., for sweeps using the dataset dir, it is 1.
        -v, --value-name: Metric name to compute statistics for. E.g., 'test/acc' or 'val/acc'.
        -s, --select: Method for selecting values ('auto', 'max', 'last'). Default is 'auto'.

    Example:
        python script_name.py -p /path/to/experiments -l 1 -v test/acc -s auto

    Note:
        - Level 0 computes the metric for experiments directly under the specified path.
        - Level 1 sweeps through subdirectories under the specified path and computes metrics.
        - 'auto' selection chooses 'last' for 'fixmatch' experiments and 'max' for others.
"""
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd


def compute_value(path: Path, value_name: str, select: str = "auto") -> pd.DataFrame:
    """
    Computes summary statistics for a given metric across multiple experiments and saves them in a file named:
    `path/{value_name}.csv`

    Args:
        path (Path): The base path containing experiment subdirectories.
        value_name (str): The name of the metric to compute statistics for.
        select (str, optional): The method for selecting values ('max', 'last', 'auto').
            If 'auto', 'last' is used for 'fixmatch' experiments and 'max' for others. Default is 'auto'.

    Returns:
        pd.DataFrame: A DataFrame containing summary statistics (mean, std, runs) for the specified metric.

    Example:
        >>> exp_path = Path('/path/to/experiments')
        >>> value_name = 'test_accuracy'
        >>> summary_df = compute_value(exp_path, value_name, select='auto')
        >>> print(summary_df)
          Experiment    Mean     STD  _Runs
        0  experiment1  85.43  1.2345     5
        1  experiment2  89.12  0.9876     5
        ...
    """
    values = []
    if select == "auto":
        if "fixmatch" in path.name.lower():
            select = "last"
        else:
            select = "max"
    for sub_path in path.iterdir():
        if sub_path.is_dir():
            file = sub_path / "metrics.csv"
            try:
                if "test" in value_name:
                    df_exp = pd.read_csv(file)
                    value = df_exp[value_name].iloc[-1]
                    values.append(value)
                else:
                    df_exp = pd.read_csv(file)
                    if select == "max":
                        value = df_exp[value_name].max()
                    elif select == "last":
                        value = df_exp[value_name].dropna().iloc[-1]
                    else:
                        raise ValueError
                    values.append(value)

            except:
                print("File not found :{}".format(file))
    values = np.array(values)
    out_dict = {
        "Experiment": str(path.parts[-1]),
        "Mean": np.round(values.mean(), 4) * 100,
        "STD": np.round(values.std(ddof=1), 4) * 100,
        "_Runs": len(values),
    }
    for key in out_dict:
        out_dict[key] = [out_dict[key]]
    df = pd.DataFrame(out_dict)
    df.to_csv(path / (value_name.replace("/", "_") + ".csv"))
    pprint(df)
    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", type=str)
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=0,
        help="Gives the amount of folders which are swept. E.g. for sweeps using the dataset dir it is 1",
    )
    parser.add_argument(
        "-v",
        "--value-name",
        type=str,
        default="test/acc",
        help="Which value should be aggregated e.g. test/acc, val/acc",
    )
    parser.add_argument(
        "-s",
        "--select",
        type=str,
        default="auto",
        help="Which value to select, either [auto, max, last]",
    )
    args = parser.parse_args()
    level = args.level
    path = args.path
    select = args.select
    path = Path(path)
    value_name = args.value_name
    if level == 0:
        compute_value(path, value_name, select)
    if level == 1:
        for sub_path in path.iterdir():
            if sub_path.is_dir():
                compute_value(sub_path, value_name, select)
    else:
        raise NotImplementedError
