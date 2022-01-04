from argparse import ArgumentParser
import os
import subprocess
from itertools import product

parser = ArgumentParser()
parser.add_argument("-c", "--cluster", action="store_true")
parser.add_argument("-d", "--debug", action="store_true")
args = parser.parse_args()

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "src/run_training_fixmatch.py")

if args.cluster:
    ex_call = "cluster_run"
    log_path = "/gpu/checkpoints/OE0612/c817h"
    subprocess.call("cluster_sync", shell=True)
else:
    ex_call = "python"
    log_path = "/home/c817h/Documents/logs_cluster"

active = ["standard"]
data = ["cifar10"]
model = ["resnet_fixmatch"]

base_path = f"{log_path}/SSL/SimCLR/cifar10"

sweep_name = "sweep_fixmatch-pretrained"

pretrained_paths = [
    f"{base_path}/2021-11-11_16:20:56.103061/checkpoints/last.ckpt",
    f"{base_path}/2021-11-15_10:29:02.475176/checkpoints/last.ckpt",
    f"{base_path}/2021-11-15_10:29:02.500429/checkpoints/last.ckpt",
]
learning_rate = [0.003]  # set according to EMAN paper
dropout_p = [0, 0.5]
seed = 12345
finetune = [True, False]
use_ema = [True, False]
small_head = [True, False]
eman = [True, False]
freeze_encoder = [False, True]
# freeze_encoder = [True]
# num_labelled = [50, 100, 250]
num_labelled = [40]
max_epochs = [2000]
# max_epochs = [200, 1000, 2000, 10000]

n_runs = 1
name_add = "_epochs-{}_labeled-{}"

base_path = f"{log_path}/SSL/SimCLR/cifar10"

experiment_name_list = []

full_iterator = product(
    active,
    data,
    model,
    learning_rate,
    finetune,
    use_ema,
    eman,
    dropout_p,
    small_head,
    num_labelled,
    max_epochs,
)

full_launches = len(list(full_iterator)) * n_runs

# this has to be initialized later again list(...)
full_iterator = product(
    active,
    data,
    model,
    learning_rate,
    finetune,
    use_ema,
    eman,
    dropout_p,
    small_head,
    num_labelled,
    max_epochs,
    freeze_encoder,
)

for run, load_pretrained_r in zip(range(n_runs), pretrained_paths):
    for i, (
        active_r,
        data_r,
        model_r,
        learning_rate_r,
        finetune_r,
        use_ema_r,
        eman_r,
        dropout_p_r,
        small_head_r,
        num_labelled_r,
        max_epochs_r,
        freeze_encoder_r,
    ) in enumerate(full_iterator):
        if eman_r and (use_ema_r is False):
            full_launches -= 1
            continue
        if finetune_r and freeze_encoder_r:
            full_launches -= 1
            continue

        if freeze_encoder_r and (small_head_r is False):
            # for training with big head lr = 0.003 works better than 0.03! and num_labels=40
            learning_rate_r *= 1

        if freeze_encoder_r and small_head_r:
            # for training with small lr = 0.03 works best than 0.03!
            learning_rate_r *= 10

        seed_exp = seed + run
        name_add_r = name_add.format(max_epochs_r, num_labelled_r)
        experiment_name = f"{sweep_name}_{data_r}_{model_r}{name_add_r}"
        configs = f"model={model_r} data={data_r} active={active_r}"
        active_args = f"++active.num_labelled={num_labelled_r}"
        model_args = f"++model.use_ema={use_ema_r} ++model.learning_rate={learning_rate_r} ++model.finetune={finetune_r}"
        model_args = f"{model_args} ++model.dropout_p={dropout_p_r} ++model.small_head={small_head_r} ++model.freeze_encoder={freeze_encoder_r}"
        model_args = f"{model_args} ++model.load_pretrained={load_pretrained_r}"
        sem_sl_args = f"++sem_sl.eman={eman_r}"
        trainer_args = f"++trainer.seed={seed_exp} ++trainer.max_epochs={max_epochs_r}"

        full_args = f"++trainer.experiment_name={experiment_name} {configs} {active_args} {model_args} {sem_sl_args} {trainer_args}"

        launch_command = f"{ex_call} {exec_path} {full_args}"
        experiment_name_list.append(experiment_name)
        print(f"Launch: {len(experiment_name_list)}/{full_launches}")
        print(launch_command)
        subprocess.call(launch_command, shell=True)
        if args.debug:
            break
    if args.debug:
        break

print(f"Num launches: {len(experiment_name_list)}/{full_launches}")
print(experiment_name_list)
