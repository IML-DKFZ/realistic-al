from argparse import ArgumentParser
import os
import subprocess
from itertools import product
import time

parser = ArgumentParser()
parser.add_argument("-c", "--cluster", action="store_true")
args = parser.parse_args()

if args.cluster:
    ex_call = "cluster_run"
    log_path = "/gpu/checkpoints/OE0612/c817h"
else:
    ex_call = "python"
    log_path = "/home/c817h/Documents/logs_cluster"

active = "standard"
data = "cifar10"
model = "wideresnet-cifar10"
early_stop = "False"
learning_rate = 0.003
seed = 12345
finetune = [True]
use_ema = [True, False]
small_head = [True, False]
acq = ["random", "entropy"]
n_runs = 3

base_path = f"{log_path}/SSL/SimCLR/cifar10"

name_add = "_fixmatch_big-head"

arguments = f"model={model} data={data} active={active}"
arguments = f"{arguments} ++model.use_ema={use_ema}"
arguments = f"{arguments} "

for i in range(n_runs):
    for acq in ["random" "entropy"]:
        seed_exp = seed + i
        experiment_name = f"{data}_acq-{active}_det-{model}_{acq}{name_add}"
        full_arg = f"++trainer.experiment_name={experiment_name} query={acq} {arguments} ++model.learning_rate={learning_rate}"
        full_arg = f"++trainer.seed={seed_exp} {full_arg}"

        execute = f"{ex_call} src/main_fixmatch.py {full_arg}"
        print(f"Execute #{i+1}:")
        print(execute)
        subprocess.call(execute, shell=True)
