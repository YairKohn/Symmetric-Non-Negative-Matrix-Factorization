import numpy as np
from math import sqrt

"""
Contains utility functions and constants
for Symmetric Non-Negative Matrix Factorization (SymNMF) python files: symnmf.py, analysis.py.
"""

EPSILON = 1e-4
MAX_ITER = 300
np.random.seed(1234)

def read_points(path):
    """
    Reads points from a file and returns them as a list of lists.
    :param path: The path to the file containing the points.
    :type path: str
    :return: A list of lists representing the points.
    :rtype: list of lists
    """
    try:
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.tolist()
    except Exception:
        print("An Error Has Occurred")
        return []

def init_H_from_W(W,n, k):
    """
    Initialize H from W using random values in the required interval.
    H ~ U(0, 2*sqrt(mean(W)/k)).
    :param W: The W matrix.
    :type W: list of lists
    :param n: The number of points.
    :type n: int
    :param k: The number of clusters.
    :type k: int
    :return: The initialized H matrix.
    :rtype: list of lists
    """    
    m = np.mean(np.array(W))
    H = np.random.uniform(0, 2*np.sqrt(m/k), (n, k)).tolist()
    return H