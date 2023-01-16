from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

import os

src_path = Path(__file__).resolve().parent.parent / "src"
# print(src_path)
# sys.path.append(src_path)

src_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
# print(src_folder)
sys.path.append(str(src_path))

# from toy_callback import ToyVisCallback
from utils.file_utils import get_experiment_df, get_experiment_configs_df
from utils.io import load_omega_conf


def compute_value(path, value_name, select="max"):
    values = []
    for sub_path in path.iterdir():
        if sub_path.is_dir():
            file = sub_path / "metrics.csv"
            try:
                hparams = load_omega_conf(sub_path / "hparams.yaml")
            except:
                pass
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
