#!/bin/bash
# source ~/.bashrc
conda_env="activeframework"

active=standard
data=cifar10
model="wideresnet-cifar10"
early_stop="False"
learning_rate=0.003
seed=12345
finetune=True
use_ema=True
model.small_head=False
# exp_path="/gpu/checkpoints/OE0612/c817h"
exp_path="/home/c817h/Documents/logs_cluster"
base_path="${exp_path}/SSL/SimCLR/cifar10"

name_add="_fixmatch_big-head"

arguments="model=${model} data=${data} active=${active}"
arguments="${arguments} ++model.use_ema=${use_ema}"
arguments="${arguments} "
i=0

ex_path="/home/c817h/code/activeframework/src"

for k in 1 2 3
do
    for acq in "random" "entropy" # "bald" "batchbald"
    do
        # echo $path
        seed_exp=$((seed + i))
        experiment_name="${data}_acq-${active}_det-${model}_${acq}${name_add}"
        full_arg="++trainer.experiment_name=${experiment_name} query=${acq} ${arguments} ++model.learning_rate=${learning_rate}"
        full_arg="trainer.seed=${seed_exp} ${full_arg}"

        execute="python src/main_fixmatch.py $full_arg"
        echo $execute
        echo
        $execute

        # execute="python ${ex_path}/main_fixmatch.py $full_arg"
        # echo $execute
        # bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate ${conda_env} && $execute"

        i=$(($i+1))
    done
done