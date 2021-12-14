#!/bin/bash
source ~/.bashrc
conda_env="activeframework"

active=standard
data=cifar10
model="resnet"
batch_size=128
momentum=0.9 # this does not change anything here - doublecheck this!
early_stop="False"
optim="sgd"
learning_rate=0.001
seed=12345
max_epochs=200
use_ema=True
# exp_path="/gpu/checkpoints/OE0612/c817h"
exp_path="/home/c817h/Documents/logs_cluster"
base_path="${exp_path}/SSL/SimCLR/cifar10"

name_add="_ssl"

arguments="model=${model} data=${data} active=${active} optim=${optim}"
arguments="${arguments} ++trainer.batch_size=${batch_size} ++model.use_ema=${use_ema}"
arguments="${arguments} ++trainer.max_epochs=${max_epochs}"

# path1="${base_path}/2021-11-11 16:20:56.103061/checkpoints/last.ckpt"
# path2="${base_path}/2021-11-15 10:29:02.475176/checkpoints/last.ckpt"
# path3="${base_path}/2021-11-15 10:29:02.500429/checkpoints/last.ckpt"

path1="${base_path}/2021-11-11_16:20:56.103061/checkpoints/last.ckpt"
path2="${base_path}/2021-11-15_10:29:02.475176/checkpoints/last.ckpt"
path3="${base_path}/2021-11-15_10:29:02.500429/checkpoints/last.ckpt"

# echo $path1
paths=( "$path1" "$path2" "$path3" )

ex_path="/home/c817h/code/activeframework/src"

i=0
for path in "${paths[@]}"
do
    for acq in "random" "kcentergreedy" "entropy" # "bald" "batchbald"
    do
        echo $path
        seed_exp=$((seed + i))
        experiment_name="${data}_acq-${active}_det-${model}_${acq}${name_add}"
        full_arg="++trainer.experiment_name=${experiment_name} query=${acq} ${arguments} ++model.learning_rate=${learning_rate}"
        # full_arg="++model.load_pretrained=\"${path}\" trainer.seed=${seed_exp} ${full_arg}"
        full_arg="++model.load_pretrained=${path} trainer.seed=${seed_exp} ${full_arg}"
        # full_arg="'++model.load_pretrained=\"${path}\"'"

        execute=" python src/main.py $full_arg"
        echo $execute
        echo 
        $execute

        # execute="python ${ex_path}/main.py $full_arg"
        # echo $execute
        # bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate ${conda_env} && $execute"

        i=$(($i+1))
    done
done


# i=0
# for path in "${paths[@]}"
# do
#     for acq in "random" "kcentergreedy" "entropy" "bald" # "batchbald"
#     do
#         # echo $path
#         seed_exp=$((seed + i))
#         experiment_name="${data}_acq-${active}_bay-${model}_${acq}${name_add}"
#         full_arg="++trainer.experiment_name=${experiment_name} query=${acq} ${arguments} ++model.learning_rate=${learning_rate}"
#         full_arg="++model.load_pretrained=\"${path}\" trainer.seed=${seed_exp} ++model.dropout_p=0.5 ${full_arg}"

#         # execute=" python src/main.py $full_arg"
#         # echo $execute
#         # # $execute

#         execute="python ${ex_path}/main.py $full_arg"
#         echo $execute
#         bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate ${conda_env} && $execute"

#         i=$(($i+1))
#     done
# done

# learning_rate=0.1
# i=0
# arguments="${arguments} ++model.freeze_encoder=True"
# name_add="_ssl-freeze"
# for path in "${paths[@]}"
# do
#     for acq in "random" "kcentergreedy" "entropy" # "bald" "batchbald"
#     do
#         seed_exp=$((seed + i))
#         experiment_name="${data}_acq-${active}_det-${model}_${acq}${name_add}"
#         full_arg="++trainer.experiment_name=${experiment_name} query=${acq} ${arguments} ++model.learning_rate=${learning_rate}" 
#         full_arg="++model.load_pretrained=\"${path}\" trainer.seed=${seed_exp} ${full_arg}"

#         # execute=" python src/main.py $full_arg"
#         # echo $execute
#         # $execute

#         execute="python ${ex_path}/main.py $full_arg"
#         echo $execute
#         bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate ${conda_env} && $execute"


#         i=$(($i+1))
#     done
# done

# # i=0
# for path in "${paths[@]}"
# do
#     for acq in "random" "kcentergreedy" "entropy" "bald" # "batchbald"
#     do
#         seed_exp=$((seed + i))
#         experiment_name="${data}_acq-${active}_bay-${model}_${acq}${name_add}"
#         full_arg="++trainer.experiment_name=${experiment_name} query=${acq} ${arguments} ++model.learning_rate=${learning_rate}" 
#         full_arg="++model.load_pretrained=\"${path}\" trainer.seed=${seed_exp} ${full_arg}"

#         # execute=" python src/main.py $full_arg"
#         # echo $execute
#         # $execute

#         execute="python ${ex_path}/main.py $full_arg"
#         echo $execute
#         bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate ${conda_env} && $execute"


#         i=$(($i+1))
#     done
# done
