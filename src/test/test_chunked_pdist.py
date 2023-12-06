# import pytest
import os
import shutil

############# Needed to execute as main ############
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_folder)

####################################################


import torch

from query.query_diversity import chunked_pdist


def test_chunked_pdist():
    x = torch.randn(5000, 10 * 1024)
    y = torch.randn(10 * 1024)
    pdist = torch.nn.PairwiseDistance(p=2)
    dist_comp = pdist(x.to("cuda:0"), y.to("cuda:0")).to("cpu")
    dist_new = chunked_pdist(x, y, max_size=1)
    print(dist_comp.shape)
    assert torch.equal(dist_comp, dist_new)


if __name__ == "__main__":
    test_chunked_pdist()
