import sys
from math import sqrt
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf
from kmeans import kmeans
import symnmf_utils


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
    H0 = symnmf_utils.init_H_from_W(W, n, k)
    H_final = symnmf.symnmf(W, H0, symnmf_utils.MAX_ITER, symnmf_utils.EPSILON)
    H_np = np.array(H_final)
    labels = np.argmax(H_np, axis=1)
    return labels


def run_kmeans(points_np, k):
    centroids = kmeans(points_np.tolist(), k, symnmf_utils.MAX_ITER)
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

    points = np.array(symnmf_utils.read_points(file_name), dtype=float)

    nmf_labels = run_symnmf(points, k)
    kmeans_labels = run_kmeans(points, k)

    nmf_score = silhouette_score(points, nmf_labels)
    kmeans_score = silhouette_score(points, kmeans_labels)

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")


if __name__ == "__main__":
    main()