#!/bin/bash 

# works relatively well
python src/run_training.py ++trainer.experiment_name=ssl_cifar ++model.learning_rate=1e-3 ++trainer.max_epochs=50 data=cifar10 model=resnet ++trainer.early_stop=False optim=sgd

python src/main.py ++trainer.experiment_name=cifar10_det_ssl_1000acq_random_lr0.001 ++model.learning_rate=1e-3 ++trainer.max_epochs=50 data=cifar10 model=resnet ++trainer.early_stop=False optim=sgd active=random


# works relatively well
python src/run_training.py ++trainer.experiment_name=ssl_cifar ++model.learning_rate=1e-3 ++trainer.max_epochs=50 data=cifar10 model=resnet ++trainer.early_stop=False optim=sgd ++model.freeze_encoder=True

# short trainings work better than long trainings, maybe this is better with stronger augmentations.

# does not converge!
python src/run_training.py ++trainer.experiment_name=ssl_cifar ++model.learning_rate=1e-2 ++trainer.max_epochs=50 data=cifar10 model=resnet ++trainer.early_stop=False optim=sgd ++model.freeze_encoder=True