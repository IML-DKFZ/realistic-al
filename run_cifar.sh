#!/bin/bash

acq_size=1000
num_labelled=1000
num_iter=10
data=cifar10
model="resnet"
batch_size=128
momentum=0.9 # this does not change anything here - doublecheck this!
early_stop="False"


arguments="model=${model} data=${data} ++active.acq_size=${acq_size} ++active.num_labelled=${num_labelled} ++active.num_iter=${num_iter} ++optim.optimizer.name=sgd ++trainer.batch_size=${batch_size} ++trainer.optim.optimizer.momentum=${momentum} ++trainer.early_stop=${early_stop}"

for i in 1 2 3
do 
    for acq in "random" "entropy" # "bald" "batchbald"
    do 
        full_arg="++trainer.experiment_name=${data}_det_${model}_${acq}_${acq_size}acq active=${acq} ${arguments}"
        # echo $full_arg
        execute="python src/main.py $full_arg"
        echo $execute
        $execute
    done
    # python src/main.py ++trainer.experiment_name=cifar10_ba_entropy_10acq active=entropy ${arguments}
    # python src/main.py ++trainer.experiment_name=cifar10_ba_bald_10acq active=bald ${arguments}
    # python src/main.py ++trainer.experiment_name=cifar10_ba_random_10acq active=random ${arguments}
    # python src/main.py ++trainer.experiment_name=cifar10_ba_batchbald_10acq active=batchbald ${arguments}
done