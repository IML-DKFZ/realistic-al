import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger


def set_seed(seed: int):
    """Set seed with modules: pytorch lightning, torch, numpy and random.
    Disable cudnn.benchmark and enables cudnn.deterministic

    Args:
        seed (int): seed set for all modules.
    """
    logger.info("SETTING GLOBAL SEED: {}".format(seed))
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
