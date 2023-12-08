# Experiment Series with the Experiment Launcher
The experiments are executed via bsub commands using the Experiment Launcher

> Information regarding datasets: For all imbalanced datasets, the default training is balanced sampling during training. These experiments are denoted with `{name}_balanced.py`

### Active Learning Experiments
Active Learning experiments are ordered by their respective dataset and training paradigms.
They are located in the `/launchers` folder and denoted as `exp_{dataset}_{training-paradigm}{info}.py`.


### Self-Supervised Pretraining
For the self-supervised models the paths in the launchers have to be set accordingly and the SimCLr trainings need to be executed.
These are located in the `/launchers` folder and denoted as `simclr_{dataset}`.

## Single Trainings

### Full Dataset Baselines
The baseline models trained on the entire dataset can be obtained with scripts in the `/launchers` folder and are named `full_{dataset}{training}.py`

### Sweeps
Launcher scripts using are located in folder: `/launchers`
- Runs:
	- CIFAR-10
		- sweep_cifar10_basic.py
		- sweep_cifar10_basic-pretrained.py
		- sweep_cifar10_fixmatch.py
	- CIFAR-100
		- sweep_cifar100_basic.py
		- sweep_cifar100_basic-pretrained.py
		- sweep_cifar100_fixmatch.py
	- CIFAR-10 LT
		- sweep_cifar100_basic_balanced.py
		- sweep_cifar100_basic-pretrained_balanced..py
		- sweep_cifar10imb_fixmatch.py
	- MIO-TCD
		- sweep_miotcd_basic_balanced.py
		- sweep_miotcd_basic-pretrained_balanced..py
		- sweep_miotcd_fixmatch.py
	- ISIC-2019
		- sweep_isic2019_basic_balanced.py
		- sweep_isic2019_basic-pretrained_balanced.py
		- sweep_isic2019_fixmatch.py
- After runs are finalized:
	- metric = `val/acc` (and `val/w_acc` for MIO-TCD and ISIC-2019)
		- for final fixmatch values also add `test/{}`
	- path = `$EXPERIMENT_ROOT/activelearning/{dataset}/sweeps`
	- execute script: `/analysis/obtain_metrics.py -l 1 -v {metric} -s {selection} -p {path}`
		- use `selection=auto` this will use for fixmatch experiments`selection=last` for others `selection=max`
- After metrics are obtained:
	- use {path} in `/analysis/results_overview.py` to visualize and interpret results of sweeps