## Place for Launchers of Experiment Rows

## Naming Scheme:
Active Iteration:
```
active_{training_type}_{data}_set-{}_{training}_{model}_acq-{}_ep-{}
```

Sweep:
```
sweep_{training_type}_{data}_lab-{trainer.num_labelled}_{model}_ep-{trainer.max_epochs}
```

### Example
Active Iteration:
```
active_basic_cifar10_set-standard_basic_resnet18_acq-bald_ep-200
```
Sweep:
```
sweep_basic-pretrained_cifar10_lab-40_resnet_fixmatch_ep-2000
```
