from argparse import ArgumentParser

from pathlib import Path
import pandas as pd
import numpy as np

from pprint import pprint


def compute_value(path, value_name, select="max"):
    values = []
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
    # df_row = [str(path.parts[-1]), values.mean(), values.std(ddof=1), len(values)]
    # pprint(out_dict)
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
        default="max",
        help="Which value to select, either [max, last]",
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
