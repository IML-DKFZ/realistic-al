# Active Learning Playground
Active Learning on Torchvision Classification Datasets

## To Do: 
- [x] Incorporate Core-Set K-Center Greedy Approach 
- [ ] Incorporate Learning Loss from https://github.com/Mephisto405/Learning-Loss-for-Active-Learning 
- [ ] Fix ugly DEVICE parameters!
- [ ] Make dataloaders deterministic
- [ ] Benchmark FP16 training
- [ ] Enable benchmark training
- [ ] Allow wider variety of data splits (random, random balanced, balanced - first k)
- [ ] Possibly add scheduler for EMA models to enforce fast learning for "shorter trainings"
- [ ] Add Self-Supervised Pretext Training into this repo for consistency
- [ ] Cleanup FixMatch Dataloader
- [ ] Reorder Config
    - [ ] Num MonteCarlo eval (k) from active to model
    - [ ]