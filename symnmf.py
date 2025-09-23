import sys
from math import sqrt
import numpy as np

# Import compiled C extension as symnmf (no alias)
import symnmf

EPSILON = 1e-4
MAX_ITER = 300
np.random.seed(1234)

def read_points(path):
    try:
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.tolist()
    except Exception:
        print("An Error Has Occurred")
        return []


def print_matrix(mat):
    for row in mat:
        print(','.join(f"{val:.4f}" for val in row))


def init_H_from_W(W,n, k):
    m = np.mean(np.array(W))
    H = np.random.uniform(0, 2*np.sqrt(m/k), (n, k)).tolist()
    return H


def parse_args(argv):
    if len(argv) != 4:
        return None
    try:
        k = int(argv[1])
    except Exception:
        return None
    return k, argv[2], argv[3]


def dispatch_goal(k, goal, file_name):
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
    parsed = parse_args(sys.argv)
    if not parsed:
        print("An Error Has Occurred")
        return
    k, goal, file_name = parsed
    if not dispatch_goal(k, goal, file_name):
        print("An Error Has Occurred")


if __name__ == '__main__':
    main()
