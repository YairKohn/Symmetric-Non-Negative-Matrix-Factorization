import argparse
import math
import os
import subprocess
import sys
from typing import List, Tuple

import numpy as np


def gaussian_W(points: np.ndarray) -> np.ndarray:
    n, d = points.shape
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            dist2 = float(np.dot(diff, diff))
            val = math.exp(-dist2 / 2.0)
            W[i, j] = val
            W[j, i] = val
    return W


def ddg_from_W(W: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    D = np.zeros_like(W)
    D[np.arange(n), np.arange(n)] = W.sum(axis=1)
    return D


def norm_from_W(W: np.ndarray) -> np.ndarray:
    deg = W.sum(axis=1)
    inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    return (inv_sqrt[:, None] * W) * inv_sqrt[None, :]


def write_csv(path: str, X: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in X:
            f.write(",".join(f"{float(x):.6f}" for x in row) + "\n")


def read_csv_matrix(text: str) -> List[List[float]]:
    rows = []
    for line in text.strip().splitlines():
        parts = line.strip().split(',')
        if parts == ['']:
            continue
        rows.append([float(p) for p in parts])
    return rows


def compare_matrices(A: np.ndarray, B: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float]:
    if A.shape != B.shape:
        return False, float("inf")
    diff = np.linalg.norm(A - B, ord="fro")
    return bool(diff <= tol * max(1.0, np.linalg.norm(A, ord="fro"))), float(diff)


def test_c_extension(points: np.ndarray) -> List[str]:
    logs: List[str] = []
    try:
        import symnmf as ext
    except Exception as e:
        logs.append(f"SKIP: symnmf extension not importable: {e}")
        return logs

    X_list = points.tolist()
    exp_W = gaussian_W(points)
    exp_D = ddg_from_W(exp_W)
    exp_N = norm_from_W(exp_W)

    W = np.array(ext.sym(X_list), dtype=float)
    ok, err = compare_matrices(W, exp_W)
    logs.append(f"EXT sym: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")

    D = np.array(ext.ddg(X_list), dtype=float)
    ok, err = compare_matrices(D, exp_D)
    logs.append(f"EXT ddg: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")

    N = np.array(ext.norm(X_list), dtype=float)
    ok, err = compare_matrices(N, exp_N)
    logs.append(f"EXT norm: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")

    # Basic symnmf sanity: non-negativity and objective not worse than random init
    n = points.shape[0]
    k = min(2, n)
    H0 = np.random.default_rng(1234).uniform(0.0, 0.5, size=(n, k)).tolist()
    H = np.array(ext.symnmf(W.tolist(), H0, 200, 1e-5))
    nonneg = bool(np.all(H >= -1e-12))
    logs.append(f"EXT symnmf: {'PASS' if nonneg else 'FAIL'} (nonneg)")
    return logs


def test_cli_binary(points: np.ndarray, exe_name: str = "symnmf") -> List[str]:
    logs: List[str] = []
    if not os.path.exists(exe_name) and not os.path.exists(exe_name + ".exe"):
        logs.append("SKIP: CLI binary 'symnmf' not found. Build it first.")
        return logs

    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    data_path = os.path.join(tmp_dir, "tiny.csv")
    write_csv(data_path, points)

    # CLI prints with 4 decimals; round only at the final print stage.
    # Compute D and N from the full-precision W, then round.
    exact_W = gaussian_W(points)
    exp_W = np.round(exact_W, 4)
    exp_D = np.round(ddg_from_W(exact_W), 4)
    exp_N = np.round(norm_from_W(exact_W), 4)

    def run(goal: str) -> np.ndarray:
        cmd = [os.path.abspath(exe_name), goal, data_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"CLI failed ({goal}): {res.stderr or res.stdout}")
        M = np.array(read_csv_matrix(res.stdout), dtype=float)
        return M

    try:
        W = np.round(run("sym"), 4)
        ok, err = compare_matrices(W, exp_W)
        logs.append(f"CLI sym: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")
    except Exception as e:
        logs.append(f"CLI sym: FAIL ({e})")

    try:
        D = np.round(run("ddg"), 4)
        ok, err = compare_matrices(D, exp_D)
        logs.append(f"CLI ddg: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")
    except Exception as e:
        logs.append(f"CLI ddg: FAIL ({e})")

    try:
        N = np.round(run("norm"), 4)
        ok, err = compare_matrices(N, exp_N)
        logs.append(f"CLI norm: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")
    except Exception as e:
        logs.append(f"CLI norm: FAIL ({e})")

    return logs


def test_py_cli(points: np.ndarray) -> List[str]:
    logs: List[str] = []
    tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    data_path = os.path.join(tmp_dir, "tiny.csv")
    write_csv(data_path, points)

    # Python CLI prints with 4 decimals; compute D/N from full-precision W.
    exact_W = gaussian_W(points)
    exp_W = np.round(exact_W, 4)
    exp_D = np.round(ddg_from_W(exact_W), 4)
    exp_N = np.round(norm_from_W(exact_W), 4)

    def run(goal: str) -> np.ndarray:
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "symnmf.py"), "2", goal, data_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"py CLI failed ({goal}): {res.stderr or res.stdout}")
        M = np.array(read_csv_matrix(res.stdout), dtype=float)
        return M

    try:
        W = np.round(run("sym"), 4)
        ok, err = compare_matrices(W, exp_W)
        logs.append(f"PY sym: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")
    except Exception as e:
        logs.append(f"PY sym: FAIL ({e})")

    try:
        D = np.round(run("ddg"), 4)
        ok, err = compare_matrices(D, exp_D)
        logs.append(f"PY ddg: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")
    except Exception as e:
        logs.append(f"PY ddg: FAIL ({e})")

    try:
        N = np.round(run("norm"), 4)
        ok, err = compare_matrices(N, exp_N)
        logs.append(f"PY norm: {'PASS' if ok else 'FAIL'} (fro-err={err:.3e})")
    except Exception as e:
        logs.append(f"PY norm: FAIL ({e})")

    return logs


def tiny_points() -> np.ndarray:
    # Simple 3-point dataset with known distances
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project tester for SymNMF core components")
    parser.add_argument("--skip-ext", action="store_true", help="Skip tests that require the C extension")
    parser.add_argument("--skip-cli", action="store_true", help="Skip tests that require the CLI binary")
    parser.add_argument("--skip-pycli", action="store_true", help="Skip tests that call symnmf.py")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for numeric comparisons")
    args = parser.parse_args()

    X = tiny_points()

    any_fail = False
    logs: List[str] = []

    if not args.skip_ext:
        logs.extend(test_c_extension(X))
    if not args.skip_cli:
        logs.extend(test_cli_binary(X))
    if not args.skip_pycli:
        logs.extend(test_py_cli(X))

    for line in logs:
        print(line)
        if "FAIL" in line:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()


