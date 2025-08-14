import sys
from math import sqrt
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf
from kmeans import kmeans

EPSILON = 1e-4
MAX_ITER = 300


def read_points(path):
    arr = np.loadtxt(path, delimiter=',')
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def init_H_from_W(W, n, k):
    # W is list of lists from C module; compute mean and init H
    np.random.seed(1234)
    W_arr = np.array(W, dtype=float)
    m = float(np.mean(W_arr))
    upper = 2.0 * sqrt(m / float(k)) if k > 0 else 0.0
    H = np.random.uniform(0.0, upper, size=(n, k))
    return H.tolist()


def labels_from_centroids(points, centroids):
    points_arr = np.array(points, dtype=float)
    centroids_arr = np.array(centroids, dtype=float)
    # distances shape: (n_points, n_centroids)
    dists = np.linalg.norm(points_arr[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    return labels


def run_symnmf(points_np, k):
    X_list = points_np.tolist()
    W = symnmf.norm(X_list)
    n = len(X_list)
    H0 = init_H_from_W(W, n, k)
    H_final = symnmf.symnmf(W, H0, MAX_ITER, EPSILON)
    H_np = np.array(H_final)
    labels = np.argmax(H_np, axis=1)
    return labels


def run_kmeans(points_np, k):
    centroids = kmeans(points_np.tolist(), k, MAX_ITER)
    labels = labels_from_centroids(points_np, centroids)
    return labels


def main():
    if len(sys.argv) != 3:
        print("Invalid Input!")
        return
    try:
        k = int(sys.argv[1])
    except Exception:
        print("Invalid Input!")
        return
    file_name = sys.argv[2]

    points = read_points(file_name)

    nmf_labels = run_symnmf(points, k)
    kmeans_labels = run_kmeans(points, k)

    nmf_score = silhouette_score(points, nmf_labels)
    kmeans_score = silhouette_score(points, kmeans_labels)

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


if __name__ == "__main__":
    main()
