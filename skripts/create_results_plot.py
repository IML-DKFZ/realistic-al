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
from utils.file_utils import get_experiment_df, get_experiment_configs_df
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

# Setting 1
# experiment_paths = [
#     "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-bald_ep-40/2022-03-09_12-02-48-360225",
#     "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-batchbald_ep-40/2022-03-09_13-00-58-016157",
#     "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-entropy_ep-40/2022-03-09_11-00-35-890630",
#     "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-entropy_ep-40/2022-03-09_11-28-28-514934",
#     # "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-kcentergreedy_ep-40/2022-03-09_11-57-01-136039",
#     # "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-kcentergreedy_ep-40/2022-03-09_11-59-54-601205",
#     "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-random_ep-40/2022-03-09_10-06-20-338102",
#     "/home/c817h/Documents/logs/activelearning/2dtoy/active_basic_toy_moons_set-toy_two_moons_bayesian_mlp_acq-random_ep-40/2022-03-09_10-33-10-595623",
# ]

# Setting 2
# hue_names = [
#     "BALD (P=0.25)",
#     "BatychBald (P=0.25)",
#     "Entropy (P=0.25)",
#     "Random (P=0.25)",
#     # "k-Center Greedy (P=0.0)",
#     # "k-Center Greedy (P=0.5)",
#     "Entropy (P=0.0)",
#     "Random (P=0.0)",
# ]
# paths = [
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-bald_ep-40",
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-batchbald_ep-40",
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-entropy_ep-40",
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-random_ep-40",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-batchbald_ep-40",
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-entropy_ep-40",
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-random_ep-40",
# ]
# base_path = "/home/c817h/Documents/logs/activelearning/toy_moons"

# Version 2
# paths = [
#     # Deep Models
#     # "basic_bayesian_mlp_deep_drop-0.25_wd-0",
#     # "basic_bayesian_mlp_deep_drop-0.25_wd-0.01",
#     # "basic_bayesian_mlp_deep_drop-0.25_wd-0.001",
#     # # "basic_bayesian_mlp_deep_drop-0.25_wd-0.1",
#     # "basic_bayesian_mlp_deep_drop-0_wd-0",
#     # "basic_bayesian_mlp_deep_drop-0_wd-0.01",
#     # "basic_bayesian_mlp_deep_drop-0_wd-0.001",
#     # # "basic_bayesian_mlp_deep_drop-0_wd-0.1",
#     #
#     # Normal Models
#     #
#     "basic_bayesian_mlp_drop-0.25_wd-0",
#     "basic_bayesian_mlp_drop-0.25_wd-0.01",
#     "basic_bayesian_mlp_drop-0.25_wd-0.001",
#     #
#     "basic_bayesian_mlp_drop-0_wd-0",
#     "basic_bayesian_mlp_drop-0_wd-0.01",
#     "basic_bayesian_mlp_drop-0_wd-0.001",
# ]
# hue_names = None
# base_path = "/home/c817h/Documents/logs/activelearning/toy_moons_sweeps"

##############
# Version 3
##############
# paths = [
#     # Normal Models dropout 0.25
#     #
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-bald_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-bald_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-batchbald_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-entropy_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-kcentergreedy_ep-12",
#     "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-random_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-variationratios_ep-12",
#     #
#     # No Dropout
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-entropy_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-kcentergreedy_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-random_ep-12",
#     # "active_basic_set-toy_two_moons_bayesian_mlp_dop-0_acq-variationratios_ep-12",
#     #
#     "active_fixmatch_set-toy_two_moons_bayesian_mlp_dop-0.25_acq-random_ep-12",
# ]
# hue_names = [
#     path.split("_")[1] + " " + path.split("-")[-2].split("_")[0] for path in paths
# ]
# base_path = "/home/c817h/Documents/logs/activelearning/toy_circles"

##############
# Version 4
##############
paths = [
    "basic_set-mnist_batchbald_double_query-bald_model-bayesian_mnist_ep-200",
    "basic_set-mnist_batchbald_double_query-batchbald_model-bayesian_mnist_ep-200",
    "basic_set-mnist_batchbald_double_query-entropy_model-bayesian_mnist_ep-200",
    "basic_set-mnist_batchbald_double_query-variationratios_model-bayesian_mnist_ep-200",
    "basic_set-mnist_batchbald_double_query-random_model-bayesian_mnist_ep-200",
]
# paths = [
#     "basic_set-mnist_batchbald_query-bald_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_query-batchbald_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_query-entropy_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_query-variationratios_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_query-random_model-bayesian_mnist_ep-200",
# ]
# paths = [
#     "basic_set-mnist_batchbald_start_query-bald_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_start_query-batchbald_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_start_query-entropy_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_start_query-variationratios_model-bayesian_mnist_ep-200",
#     "basic_set-mnist_batchbald_start_query-random_model-bayesian_mnist_ep-200",
# ]
hue_names = [path.split("query-")[1].split("_")[0] for path in paths]
# base_path = "/home/c817h/Documents/logs_cluster/activelearning/mnist"
base_path = "/home/c817h/Documents/logs_cluster/activelearning/fashion_mnist"

experiment_paths = [os.path.join(base_path, path) for path in paths]
# Note: Means over multiple different data allows simple
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # allows to use multiple arguments.
    parser.add_argument("-p", "--paths", type=str, nargs="*", default=[])
    parser.add_argument("-n", "--names", type=str, nargs="*", default=[])
    args = parser.parse_args()
    paths = args.paths
    names = args.names
    if len(args.paths) != 0:
        experiment_paths = paths
        if len(names) == paths:
            hue_names = names
        else:
            hue_names = [None for i in range(len(experiment_paths))]
    if hue_names is None:
        hue_names = [None for i in range(len(experiment_paths))]

    df = []
    for name, base_dir in zip(hue_names, experiment_paths):
        base_dir = Path(base_dir)
        # import IPython

        # IPython.embed()
        # try:
        experiment_frame = get_experiment_df(base_dir, name)

        # this is currently highly experimental -- try to get the configs for later use.
        # experiment_param_frame = get_experiment_configs_df(base_dir, name)

        df.append(experiment_frame)
        # except:
        #     for path in base_dir.iterdir():
        #         if path.is_dir():
        #             experiment_frame = get_experiment_df(path, name)
        #             df.append(experiment_frame)
        # df.append(experiment_frame)
    df = pd.concat(df)
    df.reset_index(inplace=True)

    from plotlib.performance_plots import plot_standard_dev

    fig, ax = plt.subplots()
    plot_standard_dev(ax, df, hue="Name")

    save_path = Path(visuals_folder) / "performance_plot.png"
    plt.savefig(save_path)
