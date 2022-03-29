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
sys.path.append(src_folder)

# from toy_callback import ToyVisCallback
# from utils.path_utils import visuals_folder
from utils.file_utils import get_experiment_df
from utils.path_utils import visuals_folder

# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("--path", "-p", type=str)

hue_names = ["BatchBALD", "BALD", "Entropy", "Random"]
experiment_paths = [
    "/home/c817h/Documents/logs/activelearning/mnist_bb__batchbald_10acq_final",
    "/home/c817h/Documents/logs/activelearning/mnist_bb__bald_10acq_final",
    "/home/c817h/Documents/logs/activelearning/mnist_bb__entropy_10acq_final",
    "/home/c817h/Documents/logs/activelearning/mnist_bb__random_10acq_final",
]
df = []
for name, base_dir in zip(hue_names, experiment_paths):
    base_dir = Path(base_dir)
    # import IPython

    # IPython.embed()
    experiment_frame = get_experiment_df(base_dir, name)
    df.append(experiment_frame)
df = pd.concat(df)
df.reset_index(inplace=True)

from plotlib.performance_plots import plot_standard_dev

fig, ax = plt.subplots()
plot_standard_dev(ax, df, hue="Name")

save_path = Path(visuals_folder) / "performance_plot.png"
plt.savefig(save_path)
