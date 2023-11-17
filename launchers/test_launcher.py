import subprocess
from pathlib import Path

FILEPATH = Path(__file__).resolve().parent

sweeps_standard = [
    "sweep_cifar10_basic.py",
    "sweep_cifar100_basic.py",
    "sweep_cifar10imb_basic_balanced.py",
    "sweep_isic2019_basic_balanced.py",
    "sweep_miotcd_basic_balanced.py",
]

sweeps_selfsl = [
    "sweep_cifar10_basic-pretrained.py",
    "sweep_cifar100_basic-pretrained.py",
    "sweep_cifar10imb-pretrained_basic.py",
    "sweep_isic2019_basic-pretrained_balanced.py",
    "sweep_miotcd_basic-pretrained_balanced.py",
]

sweeps_semsl = [
    "sweep_cifar10_fixmatch.py",
    "sweep_cifar100_fixmatch.py",
    "sweep_cifar10imb_fixmatch_balanced.py",
    "sweep_isic2019_fixmatch.py",
    "sweep_miotcd_fixmatch.py",
]

exp_standard = [
    "exp_cifar10_basic.py",
    "exp_cifar100_basic.py",
    "exp_cifar10imb_basic_balanced.py",
    "exp_isic2019_basic_balanced.py",
    "exp_miotdcd_basic_balanced.py",
]

exp_selfs = [
    "exp_cifar10_basic-pretrained.py",
    "exp_cifar100_basic-pretrained.py",
    "exp_cifar10imb_basic-pretrained_balanced.py",
    "exp_isic2019_basic-pretrained_balanced.py",
    "exp_miotdcd_basic-pretrained_balanced.py",
]

exp_semisl = [
    "exp_cifar10_fixmatch.py",
    "exp_cifar100_fixmatch.py",
    "exp_cifar10imb_fixmatch.py",
]
