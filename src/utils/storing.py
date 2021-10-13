from dataclasses import dataclass

import numpy as np


@dataclass
class ActiveStore:
    """Class to capture the outputs of the ActiveCallback"""

    requests: np.ndarray
    n_labelled: int
    accuracy_val: float
    accuracy_test: float
    labels: np.ndarray
