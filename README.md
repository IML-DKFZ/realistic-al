# Active Learning Playground
Active Learning on Torchvision Classification Datasets

## Experiment Laucher
Does the running of experiments for you.

Flags: 
- `-d` debug/vis mode 
	- does not run experiments but prints the launch command
	- useful to show #of runs and check if runs are selected properly
- `-c` cluster mode
	- changes the execution call to some preset value
- `--num_start {val:int}` launches runs from val
- `--num_end {val:int}` launches runs until val

## Running FixMatch Sweeps and Hyper-Parameters
Runner Scripts using the Experiment Launcher are in Folder: `launchers`
- Runs:
	- CIFAR-10
		- sweep_cifar10_fixmatch.py
		- sweep_cifar10_fixmatch-wideresnet.py (reproducibility) 
	- CIFAR-100
		- sweep_cifar100_fixmatch.py
		- sweep_cifar100_fixmatch-wideresnet.py (reproducibility)
	- CIFAR-10 LT
		- sweep_cifar10imb_fixmatch.py
	- MIO-TCD
		- sweep_miotcd_fixmatch.py
	- ISIC-2019
		- sweep_isic2019_fixmatch.py
- After runs are finalized:
	- metric = `val/acc` (and `val/w_acc` for MIO-TCD and ISIC-2019)
		- for final fixmatch values also add `test/{}`
	- path = `$EXPERIMENT_ROOT/activelearning/{dataset}/sweeps`
	- execute script: `skripts/obtain_metrics.py -l 1 -v {metric} -s last -p {path}`
- After metrics are obtained:
	- use {path} in `ipynb/results_overview.py` to visualize and interpret results



## Working on:
- Toy Experiments 
	- Understanding how AL, Sem-SL and Self-SL interact with each other
	- Build intuition for Acquisition approaches -- What do they acquire?
	- Build intuition for importance of Model performance?
	- What does a good Active Learning Strategy Capture?
- Vision Experiments:
	- Standard Baselines (Cifar10, Cifar100)
	- Further Evaluation (Fine Grained Classification) 
	- How to combine Self-SL, Sem-SL and AL for generally good performing models?
## To Do: 
- [x] Incorporate Core-Set K-Center Greedy Approach 
- [ ] Incorporate Learning Loss from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning 
- [x] Fix ugly DEVICE parameters!
- [x] Make dataloaders deterministic
- [x] Benchmark FP16 training
- [x] Enable benchmark training
- [x] Allow wider variety of data splits (random, random balanced, balanced - first k)
- [ ] Possibly add scheduler for EMA models to enforce fast learning for "shorter trainings"
- [x] Add Self-Supervised Pretext Training into this repo for consistency
- [x] Cleanup FixMatch Dataloader
- [x] Reorder Config


# Next Steps:
- [ ] Create new Models for low label according to: https://github.com/google-research/fixmatch/blob/master/fully_supervised/fs_mixup.py 
- [ ] More Datasets are on: https://github.com/linusericsson/ssl-transfer
