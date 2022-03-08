# Active Learning Playground
Active Learning on Torchvision Classification Datasets



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
