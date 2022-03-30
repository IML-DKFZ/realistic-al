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

# hue_names = ["BatchBALD", "BALD", "Entropy", "Random"]
# experiment_paths = [
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__batchbald_10acq_final",
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__bald_10acq_final",
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__entropy_10acq_final",
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__random_10acq_final",
# ]

# hue_names = ["BatchBALD", "BALD", "Entropy", "Random"]
# experiment_paths = [
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__batchbald_10acq_final",
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__bald_10acq_final",
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__entropy_10acq_final",
#     "/home/c817h/Documents/logs/activelearning/mnist_bb__random_10acq_final",
# ]
experiment_paths = [
    "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-bald_ep-40/2022-03-09_12-02-48-360225",
    "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-batchbald_ep-40/2022-03-09_13-00-58-016157",
    "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-entropy_ep-40/2022-03-09_11-00-35-890630",
    "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-entropy_ep-40/2022-03-09_11-28-28-514934",
    # "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-kcentergreedy_ep-40/2022-03-09_11-57-01-136039",
    # "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-kcentergreedy_ep-40/2022-03-09_11-59-54-601205",
    "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-random_ep-40/2022-03-09_10-06-20-338102",
    "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-random_ep-40/2022-03-09_10-33-10-595623",
]
hue_names = [
    "BALD (P=0.5)",
    "BatychBald (P=0.5)",
    "Entropy (P=0.0)",
    "Entropy (P=0.5)",
    # "k-Center Greedy (P=0.0)",
    # "k-Center Greedy (P=0.5)",
    "Random (P=0.0)",
    "Random (P=0.5)",
]
if __name__ == "__main__":
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
