#!/bin/bash 

acq_size=10
num_labelled=20
num_iter=24
data="fashion_mnist"
k=100 # Set to 100 for final experiment
early_stop="False"
max_epochs=100

name_add="_final"
seed=12345
arguments="data=${data} ++active.acq_size=${acq_size} ++active.num_labelled=${num_labelled} ++active.num_iter=${num_iter} ++active.k=${k} ++trainer.early_stop=${early_stop} ++trainer.max_epochs=${max_epochs}"
for i in 0 1 2
do 
    for acq in "random" "entropy" "bald" "batchbald"
    do 
        seed_exp=$((seed + i))
        experiment_name="${data}_bb_${model}_${acq}_${acq_size}acq${name_add}"
        full_arg="++trainer.experiment_name=${experiment_name} ++trainer.seed=${seed_exp} active=${acq} ${arguments}"
        # echo $full_arg
        execute="python src/main.py $full_arg"
        echo $execute
        $execute
    done
done