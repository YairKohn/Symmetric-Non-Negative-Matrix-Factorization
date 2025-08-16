import argparse
import sys
from typing import List, Tuple

import numpy as np
from sklearn.metrics import silhouette_score

from kmeans import kmeans as run_kmeans

try:
	import symnmf as symnmf_ext
except Exception as e:
	symnmf_ext = None


def read_points_csv(path: str) -> np.ndarray:
	arr = np.loadtxt(path, delimiter=',')
	if arr.ndim == 1:
		arr = arr.reshape(1, -1)
	return arr.astype(float)


def row_normalize_h(H: np.ndarray) -> np.ndarray:
	row_sums = H.sum(axis=1, keepdims=True)
	row_sums[row_sums == 0.0] = 1.0
	return H / row_sums


def init_h_from_w(W: List[List[float]], n: int, k: int, rng: np.random.Generator) -> List[List[float]]:
	W_arr = np.array(W, dtype=float)
	m = float(np.mean(W_arr))
	upper = 2.0 * np.sqrt(m / float(k)) if k > 0 else 0.0
	H0 = rng.uniform(0.0, upper, size=(n, k))
	return H0.tolist()


def objective_w_h(W: np.ndarray, H: np.ndarray) -> float:
	R = W - H @ H.T
	return float(np.linalg.norm(R, ord='fro') ** 2)


def run_symnmf_labels(points: np.ndarray, k: int, restarts: int, max_iter: int, epsilon: float, seed: int) -> Tuple[np.ndarray, float]:
	if symnmf_ext is None:
		raise RuntimeError("symnmf extension is not available. Build the C extension first.")
	X_list = points.tolist()
	W = symnmf_ext.norm(X_list)
	n = len(X_list)
	rng = np.random.default_rng(seed)
	best_labels = None
	best_obj = float('inf')
	for r in range(restarts):
		H0 = init_h_from_w(W, n, k, rng)
		H_final = symnmf_ext.symnmf(W, H0, max_iter, epsilon)
		H_np = np.array(H_final, dtype=float)
		obj = objective_w_h(np.array(W, dtype=float), H_np)
		if obj < best_obj:
			H_norm = row_normalize_h(H_np)
			labels = np.argmax(H_norm, axis=1)
			best_labels = labels
			best_obj = obj
	return best_labels, best_obj


def run_kmeans_labels(points: np.ndarray, k: int, max_iter: int) -> np.ndarray:
	centroids = run_kmeans(points.tolist(), k, max_iter)
	centroids_arr = np.array(centroids, dtype=float)
	dists = np.linalg.norm(points[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :], axis=2)
	labels = np.argmin(dists, axis=1)
	return labels


def evaluate(points: np.ndarray, labels: np.ndarray) -> float:
	if len(points) < 2 or len(np.unique(labels)) < 2:
		return float('nan')
	return float(silhouette_score(points, labels))


def parse_k_values(args: argparse.Namespace) -> List[int]:
	if args.k is not None:
		return [args.k]
	if args.ks is not None:
		return [int(x) for x in args.ks.split(',') if x.strip()]
	start = 2 if args.kmin is None else args.kmin
	end = 8 if args.kmax is None else args.kmax
	return list(range(start, end + 1))


def print_table(rows: List[Tuple[int, float, float]]) -> None:
	print("k, silhouette_symnmf, silhouette_kmeans")
	for k, s_sym, s_km in rows:
		ss = "" if np.isnan(s_sym) else f"{s_sym:.4f}"
		sk = "" if np.isnan(s_km) else f"{s_km:.4f}"
		print(f"{k},{ss},{sk}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Tester for SymNMF vs K-means on CSV data")
	parser.add_argument("file", type=str, help="Path to CSV data (no header)")
	parser.add_argument("--k", type=int, default=None, help="Single k to evaluate")
	parser.add_argument("--ks", type=str, default=None, help="Comma-separated list of k values (e.g., 2,3,4)")
	parser.add_argument("--kmin", type=int, default=None, help="Range start for k (inclusive)")
	parser.add_argument("--kmax", type=int, default=None, help="Range end for k (inclusive)")
	parser.add_argument("--restarts", type=int, default=10, help="SymNMF random restarts")
	parser.add_argument("--max_iter", type=int, default=1000, help="SymNMF/K-means max iterations")
	parser.add_argument("--epsilon", type=float, default=1e-5, help="SymNMF convergence tolerance")
	parser.add_argument("--seed", type=int, default=1234, help="Random seed for SymNMF init")
	args = parser.parse_args()

	points = read_points_csv(args.file)
	k_values = parse_k_values(args)
	rows = []
	for k in k_values:
		try:
			labels_sym, _ = run_symnmf_labels(points, k, args.restarts, args.max_iter, args.epsilon, args.seed)
			sym_score = evaluate(points, labels_sym)
		except Exception as e:
			print(f"[warn] SymNMF failed for k={k}: {e}")
			sym_score = float('nan')
		labels_km = run_kmeans_labels(points, k, args.max_iter)
		km_score = evaluate(points, labels_km)
		rows.append((k, sym_score, km_score))
	print_table(rows)


if __name__ == "__main__":
	main()


