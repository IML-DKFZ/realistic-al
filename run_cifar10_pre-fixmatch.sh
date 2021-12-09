#!/bin/bash
source ~/.bashrc
conda_env="activeframework"

active=standard
data=cifar10
model="resnet_fixmatch"
early_stop="False"
learning_rate=0.001
seed=12345
finetune=True
use_ema=True
exp_path="/gpu/checkpoints/OE0612/c817h"
# exp_path="/home/c817h/Documents/logs_cluster"
base_path="${exp_path}/SSL/SimCLR/cifar10"

name_add="_fixmatch_ssl_finetune_big-head"

arguments="model=${model} data=${data} active=${active}"
arguments="${arguments} ++model.use_ema=${use_ema} ++model.finetune=${finetune}"
arguments="${arguments} "
i=0
path1="${base_path}/2021-11-11 16:20:56.103061/checkpoints/last.ckpt"
path2="${base_path}/2021-11-15 10:29:02.475176/checkpoints/last.ckpt"
path3="${base_path}/2021-11-15 10:29:02.500429/checkpoints/last.ckpt"

# echo $path1
paths=( "$path1" "$path2" "$path3" )

ex_path="/home/c817h/code/activeframework/src"

for path in "${paths[@]}"
do
    for acq in "random" "entropy" # "bald" "batchbald"
    do
        # echo $path
        seed_exp=$((seed + i))
        experiment_name="${data}_acq-${active}_det-${model}_${acq}${name_add}"
        full_arg="++trainer.experiment_name=${experiment_name} query=${acq} ${arguments} ++model.learning_rate=${learning_rate}"
        full_arg="++model.load_pretrained=\"${path}\" trainer.seed=${seed_exp} ${full_arg}"

        execute="python src/main_fixmatch.py $full_arg"
        # echo $execute
        # echo
        # $execute

        execute="python ${ex_path}/main_fixmatch.py $full_arg"
        echo $execute
        bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate ${conda_env} && $execute"

        i=$(($i+1))
    done
done