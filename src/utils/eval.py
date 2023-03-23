from typing import Union, Optional

from scipy import stats
import pandas as pd
import numpy as np


def get_aubc(
    performance: np.ndarray,
    x: Union[None, np.ndarray] = None,
    dx: Optional[float] = None,
) -> float:
    """Computes the Area Under the Budget Curve following:
    https://www.ijcai.org/proceedings/2021/0634.pdf
    Default dx makes integral go from 0 to 1
    
    Args:
        performance (np.ndarray): value under which budget should be computed
        x (Union[None, np.ndarray], optional): gives points over which to integrate. Defaults to None.
        dx (float, optional): difference of values. Defaults to None.
    """

    if dx is None:
        # simulated integral goes from 0 to 1
        dx = 1 / (len(performance) - 1)
    return np.trapz(performance, x, dx).item()


def compute_pairwise_matrix(df: pd.DataFrame, value: str = "test_acc"):
    """
    Computes and plots pairwise comparison matrix for active learning experiments.

    Code is adapted from: https://github.com/JordanAsh/badge/blob/master/scripts/agg_results.py
    We assume here that we are neven in the saturation area.
    --> Performance <= 0.99*(full dataset performance).

    Args:
        df (pd.DataFrame): _description_
        value (str, optional): _description_. Defaults to "test_acc".
    """
    algs = df["Query Method"].unique()
    algs.sort()
    matrix = {}
    for a1 in algs:
        matrix[a1] = {}
        for a2 in algs:
            matrix[a1][a2] = 0
    nExperiments = len(df["num_samples"].unique())
    for num_sample in df["num_samples"].unique():
        for alg1 in algs:
            for alg2 in algs:
                if alg1 == alg2:
                    continue
                res1 = df[df["Query Method"] == alg1]
                res2 = df[df["Query Method"] == alg2]
                exp1 = res1[res1["num_samples"] == num_sample][value].values
                exp2 = res2[res2["num_samples"] == num_sample][value].values
                n1 = len(exp1)
                n2 = len(exp2)
                if (n1 <= 1) or (n2 <= 1):
                    continue

                n = min(n1, n2)
                z = exp1[:n] - exp2[:n]
                mu = np.mean(z)
                t, pval = stats.ttest_1samp(z, 0.0)
                if mu < 0 and pval < 0.05:
                    matrix[alg1][alg2] += 1.0 / nExperiments

    return matrix
