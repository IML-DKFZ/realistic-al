<!-- # Active-Study -->

# Realistic-AL

##  *Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment*
> Official Benchmark Implementation
<p align="">
    <!-- <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/IML-DKFZ/fd-shifts/pytest.yml?branch=main&label=tests"> -->
    <a href="https://github.com/IML-DKFZ/realistic-al/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/IML-DKFZ/realistic-al">
    </a>
    <!-- <a href="https://github.com/IML-DKFZ/fd-shifts/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/IML-DKFZ/fd-shifts.svg">
    </a>
    <a href="https://zenodo.org/badge/latestdoi/570145779"><img src="https://zenodo.org/badge/570145779.svg" alt="DOI"></a> -->
</p>

---

### Abstract
> Active Learning (AL) aims to reduce the labeling burden by interactively selecting the most informative samples from a pool of unlabeled data. While there has been extensive research on improving AL query methods in recent years, some studies have questioned the effectiveness of AL compared to emerging paradigms such as semi-supervised (Semi-SL) and self-supervised learning (Self-SL), or a simple optimization of classifier configurations. Thus, today’s AL literature presents an inconsistent and contradictory landscape, leaving practitioners uncertain about whether and how to use AL in their tasks. In this work, we make the case that this inconsistency arises from a lack of systematic and realistic evaluation of AL methods. Specifically, we identify five key pitfalls in the current literature that reflect the delicate considerations required for AL evaluation. Further, we present an evaluation framework that overcomes these pitfalls and thus enables meaningful statements about the performance of AL methods. To demonstrate the relevance of our protocol, we present a large-scale empirical study and benchmark for image classification spanning various data sets, query methods, AL settings, and training paradigms. Our findings clarify the inconsistent picture in the literature and enable us to give hands-on recommendations for practitioners.

<p align="center">
    <figure class="image">
        <img src="./docs/assets/al_loop.png">
        <figcaption style="font-size: small;">
        <!-- There is no consensus regarding the performance gain of AL methods over random sampling in the literature, especially with regard to the cold start problem and orthogonal developments s.a. self-supervised and semi-supervised learning.  -->
        As practitioners need to rely on the performance gains estimated in studies to make an informed choice whether to employ AL or not, as an evaluation would require additional label effort which defeats the purpose of using AL (validation paradox).
        Therefore the evaluation needs to test AL methods with regard to the following requirements: 1) Generalization across varying data distributions, 2) robustness with regard to design choices of an AL pipeline 3), performance gains persist in combination with orthogonal approaches (e.g. Self-SL, Semi-SL).<br>
		This benchmark aims at solving these issues by improving the evaluation upon 5 concrete pitfalls in the literature (shown in action the figure above): <br>
        P1: Lack of evaluated data distribution settings. 
        P2: Lack of evaluated starting budgets.
        P3: Lack of evaluated query sizes.
        P4: Neglection of classifier configuration.
        P5: Neglection of alternative training paradigms.
        </figcaption>
    </figure>
</p>

## Citing This Work

If you use Realistic-AL please cite our [paper](https://arxiv.org/abs/2301.10625)

```bibtex
@inproceedings{
luth2023navigating,
title={Navigating the Pitfalls of Active Learning Evaluation: A Systematic Framework for Meaningful Performance Assessment},
author={Carsten Tim L{\"u}th and Till J. Bungert and Lukas Klein and Paul F Jaeger},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=Dqn715Txgl}
}
```

## Table Of Contents
---
<!--toc:start-->

- TOC
{:toc}


<!--toc:end-->
---

## Installation

**Realistic-AL requires Python version 3.8 or later.** It is recommended to
install FD-Shifts in its own environment (venv, conda environment, ...).

1. **Install an appropriate version of [PyTorch](https://pytorch.org/).** Check
   that CUDA is available and that the CUDA toolkit version is compatible with
   your hardware. The currently necessary version of
   [pytorch is v.1.12.0](https://pytorch.org/get-started/previous-versions/#v1120).
   Testing and Development was done with the pytorch version using CUDA 11.3.

2. **Install Realistic-AL.** This will pull in all dependencies including some
   version of PyTorch, it is strongly recommended that you install a compatible
   version of PyTorch beforehand.
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure 
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

To use `Realistic-AL` you need to:
1. set two environment variables described below
2. you may have to go through the code and change global variables which are highlighted with ### RUNNING ###

Set up of the environment variables.

```bash
export EXPERIMENT_ROOT=/absolute/path/to/your/experiments
export DATA_ROOT=/absolute/path/to/datasets
```

Alternatively, you may write them to a file and source that before running
`Realistic-AL`, e.g.

```bash
mv example.env .env
```

Then edit `.env` to your needs and run

```bash
source .env
```


### Data Folder Requirements

Only `$DATA_ROOT` has to be set and the data will be automatically downloaded in the corresponding folders.

### Active Learning Experiments
Active Learning experiments are ordered by their respective dataset and training paradigms.
They are located in the `/launchers` folder and denoted as `exp_{dataset}_{training-paradigm}{info}.py`.

For the self-supervised models the paths in the launchers have to be set accordingly and the SimCLr trainings need to be executed.
These are located in the `/launchers` folder and denoted as `simclr_{dataset}`.


### Standard Model Training (100% labeled Dataset)
The baseline models trained on the entire dataset can be obtained with scripts in the `/launchers` folder and are named `full_{dataset}{training}.py`



### Analysis
The implemented analysis can be found in the folder `/analysis/` and consists of:
1. Standard performance vs. labeled data plots
2. Area Under Budget Curve (AUBC)
3. Pairwise Penalty Matrices (PPM)

The main script for the analysis is located in:
`/analysis/plot.py`

To execute the analysis on your own device the results path will need to be changed in the script.
Also all values for the AUBC plots are read out in this function.
Further it requires to have all of the models that are trained on the entire dataset to be present in the default version.
For this execute: `/analysis/obtain_metrics.py -l 1 -v {metric} -s {selection} -p {path_to_full_trained_models}`

A newly implemented Query Method needs to also be introduced here.

The AUBC values and PPMS can be found in the respective jupyter notebooks.


### Sweeps
Runner Scripts using the Experiment Launcher are in Folder: `/launchers`
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
		- use `selection=auto` this will use for fixmatch experiments`selection=last` for others `selection=max`
- After metrics are obtained:
	- use {path} in `/analysis/results_overview.py` to visualize and interpret results of sweeps


## How to Integrate Your own Query Methods, Datasets, Trainings & Models

### Query Method
To add your own custom query method familiarize yourself with the class QuerySampler in `/src/query/query.py`.

For uncertainty based query methods operating purely on model outputs and ranking of specific scores s.a. Entropy, BALD, check out the examples in `/src/query_uncertainty`.

For diversity based query methods that require some form of intermediate representations s.a. Core-Set or BADGE, check out the examples in `/src/query_diversity.py`

### Datasets
To add a new dataset please check out the class BaseDataModule in `/src/data/base_datamodule.py`

### Trainings
To add a new training strategy check out the class AbstractClassifier in `/src/models/abstract_classifier.py` and its corresponding inheritors.

You also might have to add a new trainer class for this (see `/src/trainer.py` and `/src/trainer_fix.py`)

Finally you would need to add a run_training_{}.py and main_{}.py


## Acknowledgements

<p align="center">
  <img src="https://polybox.ethz.ch/index.php/s/I6VJEPrCDW9zbEE/download" width="190"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="91"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Deutsches_Krebsforschungszentrum_Logo.svg/1200px-Deutsches_Krebsforschungszentrum_Logo.svg.png" width="270">
</p>
