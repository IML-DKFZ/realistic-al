# from typing import Generator, Optional, Sequence, Union

# import numpy as np
# import pytorch_lightning as pl
# import torch
# from .random_fixed_length_sampler import RandomFixedLengthSampler
# from torch.utils.data import DataLoader, Dataset, Subset, random_split

# from .active import ActiveLearningDataset
# from .utils import activesubset_from_subset, ActiveSubset, seed_worker

# from .transformations import get_transform
"""This Module contains the Code to create toy datasets in numpy arrays."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons, make_circles, make_checkerboard, make_blobs


def generate_hypercube_data(n_samples, noise=0.3, n_dim=2, seed=12345):
    """Generate a random n-class classification problem based on hypercubes in n_dim.

    For more Information about Data Generation process see:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
    """
    X, y = make_classification(
        n_features=n_dim,
        n_samples=n_samples,
        random_state=seed,
        n_clusters_per_class=1,
        n_informative=n_dim - 1,
        n_redundant=1,
    )
    X += np.random.randn(*X.shape) * noise
    return X, y


def generate_blob_data(n_samples, noise=1, n_dim=2, centers=3, seed=12345):
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_dim,
        centers=centers,
        random_state=seed,
        cluster_std=noise,
    )
    return X, y


def generate_moons_data(n_samples, noise=0.15, seed=12345):
    X, y = make_moons(n_samples=n_samples, shuffle=True, random_state=seed, noise=noise)
    return X, y


def generate_circles_data(n_samples, noise=0.15, seed=12345, factor=0.5):
    X, y = make_circles(
        n_samples, shuffle=True, random_state=seed, noise=noise, factor=factor
    )
    return X, y

# Generating Checkerboard Data is super interesting, but needs to be implemented!
# def generate_checkerboard_data(n_samples, n_dim, seed=12345):
#     X, y = make_checkerboard()


def merge_labels(y, num_labels=2):
    """
    Merges the labels given with y so that the returned labels are in the range of num_labels.
    """
    labels_old = np.unique(y)
    assert len(labels_old.shape) == 1  # labels need to be singular integers!
    assert (
        len(labels_old) % num_labels == 0
    )  # The number of clusters must be equal to the final amount of clusters!
    label_dict = dict()
    for i, label in enumerate(labels_old):
        label_dict[label] = i % num_labels

    map_fct = lambda x: label_dict[x]

    y_new = np.array(list(map(map_fct, y)))

    return y_new


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_samples = 200
    datasets = {
        "hypercube": generate_hypercube_data(n_samples=n_samples),
        "blob_3": generate_blob_data(n_samples=n_samples, centers=3),
        "blob_4": generate_blob_data(n_samples=n_samples, centers=4),
        "moons": generate_moons_data(n_samples=n_samples),
        "circles": generate_circles_data(n_samples=n_samples),
    }

    for dataname, data in datasets.items():
        print("Plotting Data Type {}".format(dataname))
        # TODO: maybe make a check whether this works y can be CENTERS!
        try:
            X, y = data
            if len(y.shape) != 1:
                print(y.shape)
                print(y)
                # Filter here after unique y values seen from axis 0
                # use np.uniqe and a loop or np.where for this!
                raise NotImplementedError
            print("Data Shape: {}".format(X.shape))
            print("Label Shape: {}".format(y.shape))

            plt.scatter(X[:, 0], X[:, 1], c=y)
            plt.savefig("toy_{}.png".format(dataname))
            plt.cla()
            plt.clf()

            if dataname == "blob_4":
                plt.scatter(X[:, 0], X[:, 1], c=merge_labels(y, num_labels=2))
                plt.savefig("toy_{}_split.png".format(dataname))
                plt.cla()
                plt.clf()

            print("Succesful on Data Type {}".format(dataname))
        except Exception as e:
            print(e)
            print("Unsuccesful on Data Type {}".format(dataname))

    for dataname, data in datasets.items():
        print(
            "Mean and Std for Data Type {}".format(dataname)
        )
        X, y = data
        print("Mean: {}".format(X.mean(axis=0)))
        print("Std: {}".format(X.std(axis=0)))
    