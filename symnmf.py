import sys
from math import sqrt
import numpy as np
import symnmf

# Algorithm parameters
np.random.seed(1234)  # Fixed seed for reproducible results
EPSILON = 1e-4        # Convergence threshold for SymNMF iterations
MAX_ITER = 300        # Maximum number of iterations for SymNMF

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


def print_matrix(mat):
    """
    Prints a matrix in the specified format with 4 decimal places..
    :param mat: The matrix to print.
    :type mat: list of lists
    :return: void
    """
    for row in mat:
        print(','.join(f"{val:.4f}" for val in row))


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


def parse_args(argv):
    """
    Parses the arguments and returns the number of clusters, the goal, and the file name.
    :param argv: The arguments.
    :type argv: list of strings
    :return: The number of clusters, the goal, and the file name.
    :rtype: tuple of int, str, str
    """
    if len(argv) != 4:
        return None
    try:
        k = int(argv[1])
    except Exception:
        return None
    goal = argv[2]
    filename = argv[3]
    if not filename.endswith('.txt'):  # check filename extension
        return None
    
    return k, goal, filename

def dispatch_goal(k, goal, file_name):
    """
    Dispatches the goal and prints the result.
    :param k: The number of clusters.
    :type k: int
    :param goal: The goal.
    :type goal: str
    :param file_name: The name of the file containing the points.
    :type file_name: str
    :return: True if the goal was dispatched successfully, False otherwise.
    :rtype: bool
    """
    X = read_points(file_name)
    n = len(X)
    if goal == 'sym':
        print_matrix(symnmf.sym(X))
        return True
    if goal == 'ddg':
        print_matrix(symnmf.ddg(X))
        return True
    if goal == 'norm':
        print_matrix(symnmf.norm(X))
        return True
    if goal == 'symnmf':
        W = symnmf.norm(X)
        H0 = init_H_from_W(W,n, k)
        H_final = symnmf.symnmf(W, H0, MAX_ITER, EPSILON)
        print_matrix(H_final)
        return True
    return False


def main():
    """
    performs symNMF (symmetric Non-negative Matrix Factorization) and prints the result
    """
    parsed = parse_args(sys.argv)
    if not parsed:
        print("An Error Has Occurred")
        return
    k, goal, file_name = parsed
    if not dispatch_goal(k, goal, file_name):
        print("An Error Has Occurred")


if __name__ == '__main__':
    main()
