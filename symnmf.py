import sys
from math import sqrt
import numpy as np

# Import compiled C extension as symnmf (no alias)
import symnmf

EPSILON = 1e-4
MAX_ITER = 300


def read_points(path):
    arr = np.loadtxt(path, delimiter=',')
    # Ensure 2D shape even for single row
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.tolist()


def print_matrix(mat):
    for row in mat:
        print(','.join(f"{val:.4f}" for val in row))


def init_H_from_W(W, n, k):
    np.random.seed(1234)
    W_arr = np.array(W, dtype=float)
    m = float(np.mean(W_arr))
    upper = 2.0 * sqrt(m / float(k)) if k > 0 else 0.0
    H = np.random.uniform(0.0, upper, size=(n, k))
    return H.tolist()


def main():
    if len(sys.argv) != 4:
        print("Invalid Input!")
        return

    # CLI: k goal file_name
    try:
        k = int(sys.argv[1])
    except Exception:
        print("Invalid Input!")
        return

    goal = sys.argv[2]
    file_name = sys.argv[3]

    X = read_points(file_name)
    n = len(X)

    if goal == 'sym':
        A = symnmf.sym(X)
        print_matrix(A)
        return

    if goal == 'ddg':
        D = symnmf.ddg(X)
        print_matrix(D)
        return

    if goal == 'norm':
        W = symnmf.norm(X)
        print_matrix(W)
        return

    if goal == 'symnmf':
        # Build normalized similarity W, then initialize H and run
        W = symnmf.norm(X)
        H0 = init_H_from_W(W, n, k)
        H_final = symnmf.symnmf(W, H0, MAX_ITER, EPSILON)
        print_matrix(H_final)
        return

    print("Invalid Input!")


if __name__ == '__main__':
    main()
