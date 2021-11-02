#!/bin/bash 

# conda activate lidra

# python src/main.py ++trainer.experiment_name=mnist_ba_entropy active=entropy 
# python src/main.py ++trainer.experiment_name=mnist_ba_bald active=bald 
# python src/main.py ++trainer.experiment_name=mnist_ba_random active=random
# for i in 1 2 3 4 5 
# do 
#     # python src/main.py ++trainer.experiment_name=mnist_ba_batchbald_20acq active=batchbald
#     python src/main.py ++trainer.experiment_name=mnist_ba_entropy_20acq active=entropy ++active.acq_size=20
#     python src/main.py ++trainer.experiment_name=mnist_ba_bald_20acq active=bald ++active.acq_size=20
#     python src/main.py ++trainer.experiment_name=mnist_ba_random_20acq active=random ++active.acq_size=20

# done
acq_size=10
num_labelled=20
num_iter=10
data="mnist"
k=50 # Set to 100 for final experiment
early_stop="False"
max_epochs="100"

name_add="_nostop"

arguments="data=${data} ++active.acq_size=${acq_size} ++active.num_labelled=${num_labelled} ++active.num_iter=${num_iter} ++active.k=${k} ++trainer.early_stop=${early_stop} ++trainer.max_epochs=${max_epochs}"
# echo $arguments
for i in 1 
do 
    for acq in "random" #"entropy" "bald" "batchbald"
    do 
        experiment_name="${data}_bb_${model}_${acq}_${acq_size}acq${name_add}"
        full_arg="++trainer.experiment_name=${experiment_name} active=${acq} ${arguments}"
        # echo $full_arg
        execute="python src/main.py $full_arg"
        echo $execute
        $execute
    done
    # python src/main.py ++trainer.experiment_name=bb_ba_entropy_10acq active=entropy ${arguments}
    # python src/main.py ++trainer.experiment_name=bb_ba_bald_10acq active=bald ${arguments}
    # python src/main.py ++trainer.experiment_name=bb_ba_random_10acq active=random ${arguments}
    # python src/main.py ++trainer.experiment_name=bb_ba_batchbald_10acq active=batchbald ${arguments}
done