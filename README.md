<!-- # Active-Study -->
<p align="center">
    <!-- <img src="./docs/fd_shifts_logo.svg"> -->
	LOGO
    <br/>
</p>

<p align="center">
    <!-- <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/IML-DKFZ/fd-shifts/pytest.yml?branch=main&label=tests"> -->
    <!-- <a href="https://github.com/IML-DKFZ/fd-shifts/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/IML-DKFZ/fd-shifts">
    </a>
    <a href="https://github.com/IML-DKFZ/fd-shifts/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/IML-DKFZ/fd-shifts.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/570145779"><img src="https://zenodo.org/badge/570145779.svg" alt="DOI"></a> -->
</p>

---

> ABSTRACT

<p align="center">
    <figure class="image">
        <img src="./docs/assets/al_loop.png">
        <figcaption style="font-size: small;">
        In Active Learning the following parameters are often not thoroughly evaluated which heavily influence the gain of Active Learning.
		In our experiments we heavily focus on these parameters to correct common pitfalls (P1-P5) in evaluation which lead to results that are hard to interpret.
        </figcaption>
    </figure>
</p>

## Citing This Work

If you use NAME please cite our [paper](https://openreview.net/pdf?id=YnkGMIh0gvX)

```bibtex
CITATION
```

## Table Of Contents

<!--toc:start-->

- [Installation](#installation)
- [How to Integrate Your Own Usecase](#how-to-integrate-your-own-query-methods-datasets--models)
	- [Project Structue](#project-structure)
- [Usage](#usage)
  - [Data Folder Requirements](#data-folder-requirements)
  - [Training](#training)
  - [Analysis](#analysis)
  - [Sweeps](#sweeps)
- [Acknowledgements](#acknowledgements)


<!--toc:end-->

## Installation

**NAME requires Python version 3.8 or later.** It is recommended to
install FD-Shifts in its own environment (venv, conda environment, ...).

1. **Install an appropriate version of [PyTorch](https://pytorch.org/).** Check
   that CUDA is available and that the CUDA toolkit version is compatible with
   your hardware. The currently necessary version of
   [pytorch is v.1.12.0](https://pytorch.org/get-started/previous-versions/#v1120).
   Testing and Development was done with the pytorch version using CUDA 11.3.

2. **Install NAME.** This will pull in all dependencies including some
   version of PyTorch, it is strongly recommended that you install a compatible
   version of PyTorch beforehand.
   ```bash
   pip install -r requirements.txt
   ```

## How to Integrate Your own Query Methods, Datasets & Models
TODO

### Project Structure 
```
├── analysis # analysis & notebooks
│   └── plots # plots
├── launchers # launchers for experiments
├── ssl # simclr training 
│   └── config # configs for simclr
└── src
    ├── config
    ├── data # everything data
    ├── models # pl.Lightning models 
    │   ├── callbacks # lightning callbacks
    │   └── networks # model architecture
    ├── plotlib # plotlib
    ├── query # query method
    │   └── batchbald_redux # batchbald from BlackHC
    ├── test # tests
    │   └── data 
    └── utils 
```

## Usage

To use `NAME` you need to set the following environment variables

```bash
export EXPERIMENT_ROOT=/absolute/path/to/your/experiments
export DATA_ROOT=/absolute/path/to/datasets
```

Alternatively, you may write them to a file and source that before running
`NAME`, e.g.

```bash
mv example.env .env
```

Then edit `.env` to your needs and run

```bash
source .env
```

### Data Folder Requirements

Only `$DATA_ROOT` has to be set and the data will be automatically downloaded in the corresponding folders.

### Training
#### Active Learning Experiments
Active Learning experiments are ordered by their respective dataset and training paradigms.
They are situated in the launchers folder and denoted as `exp_{dataset}_{training-paradigm}{info}.py`.

For the self-supervised models the paths in the launchers have to be set accordingly and the SimCLr trainings need to be executed.
The are situated in the launchers folder and denoted as `simclr_{dataset}`.

#### Experiment Laucher
Does the running of experiments for you.

Flags: 
- `-d` debug/vis mode 
	- does not run experiments but prints the launch command
	- useful to show #of runs and check if runs are selected properly
- `-b` submission mode
	- can define custom bsub commands
- `--num_start {val:int}` launches runs from val
- `--num_end {val:int}` launches runs until val




### Analysis

TBD


### Sweeps
Runner Scripts using the Experiment Launcher are in Folder: `launchers`
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
		- sweep_isic2019_fixmatch.py
		- sweep_isic2019_fixmatch.py
		- sweep_isic2019_fixmatch.py
- After runs are finalized:
	- metric = `val/acc` (and `val/w_acc` for MIO-TCD and ISIC-2019)
		- for final fixmatch values also add `test/{}`
	- path = `$EXPERIMENT_ROOT/activelearning/{dataset}/sweeps`
	- execute script: `analysis/obtain_metrics.py -l 1 -v {metric} -s {selection} -p {path}`
		- for fixmatch experiments use `selection=last` for others use `selection=max`
- After metrics are obtained:
	- use {path} in `analysis/results_overview.py` to visualize and interpret results

## Acknowledgements

<br>

<p align="center">
  <img src="https://polybox.ethz.ch/index.php/s/I6VJEPrCDW9zbEE/download" width="190"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="91"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Deutsches_Krebsforschungszentrum_Logo.svg/1200px-Deutsches_Krebsforschungszentrum_Logo.svg.png" width="270">
</p>
