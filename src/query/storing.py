from dataclasses import dataclass

import numpy as np


# TODO: This class requires a major Rework!
@dataclass
class ActiveStore:
    """Class to capture the outputs of the ActiveCallback"""

    requests: np.ndarray  # indices of data to be requested
    n_labelled: int  # how many datapoints are labelled
    accuracy_val: float  # accuracy on validation set
    accuracy_test: float  # accuracy on test set
    labels: np.ndarray  # what are the requested labels
