from pathlib import Path
import sys
import matplotlib.pyplot as plt

import os

src_path = Path(__file__).resolve().parent.parent / "src"
# print(src_path)
# sys.path.append(src_path)

src_folder = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
# print(src_folder)
sys.path.append(src_folder)

from toy_callback import ToyVisCallback
from utils.path_utils import visuals_folder
from utils.file_utils import get_experiment_dicts

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", "-p", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    experiment_path = Path(args.path)
    dictlist = get_experiment_dicts(experiment_path)

    visuals_folder = Path(visuals_folder)
    fig, axs = ToyVisCallback.fig_full_vis_2d(dictlist)
    fig.tight_layout()
    plt.savefig(visuals_folder / "full_vis.png")
