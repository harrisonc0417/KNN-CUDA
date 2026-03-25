"""
benchmark.py  --  KNN-CUDA vs sklearn.neighbors performance comparison

Generates synthetic datasets and measures wall-clock time and accuracy/MSE
for both the CUDA implementation (knn.py) and sklearn's brute-force CPU KNN.

Usage
-----
    python3 benchmark.py                  # full sweep (10K → 1M training points)
    python3 benchmark.py --quick          # two small configs, fast smoke test
    python3 benchmark.py --n-train 500000 --n-query 20000 --n-features 128 --k 15

Requirements
------------
    make libknn.so
    pip install -r requirements.txt
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from knn import KNNClassifier, KNNRegressor


# ============================================================
# Result container
# ============================================================

@dataclass
class BenchResult:
    label:    str
    time_s:   float
    accuracy: Optional[float] = None   # classification only
    mse:      Optional[float] = None   # regression only


# ============================================================
# Data generators
# ============================================================

def _make_blobs(
    n_train: int,
    n_query: int,
    n_features: int,
    n_classes: int,
    seed: int,
    feat_dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian blobs, one per class, well-separated in feature space."""
    rng     = np.random.default_rng(seed)
    centres = rng.standard_normal((n_classes, n_features)) * 5.0

    def _blob(n):
        labels = rng.integers(0, n_classes, size=n)
        X      = centres[labels] + rng.standard_normal((n, n_features)) * 0.8
        return X.astype(feat_dtype), labels.astype(np.int32)

    train_X, train_y = _blob(n_train)
    query_X, query_y = _blob(n_query)
    return train_X, train_y, query_X, query_y


def _make_regression(
    n_train: int,
    n_query: int,
    n_features: int,
    seed: int,
    feat_dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """y = sin(X · proj) + noise, where proj is a random unit vector."""
    rng  = np.random.default_rng(seed)
    proj = rng.standard_normal(n_features).astype(np.float32)
    proj /= np.linalg.norm(proj)

    def _data(n):
        X = rng.standard_normal((n, n_features)).astype(feat_dtype)
        # Compute targets in float32 regardless of feat_dtype — sin() over
        # float16 features loses too much precision in the target values.
        y = (np.sin(X.astype(np.float32) @ proj)
             + rng.standard_normal(n) * 0.05).astype(np.float32)
        return X, y

    train_X, train_y = _data(n_train)
    query_X, query_y = _data(n_query)
    return train_X, train_y, query_X, query_y


# ============================================================
# Benchmark runners
# ============================================================

def bench_classification(
    n_train: int,
    n_query: int,
    n_features: int,
    k: int,
    n_classes: int = 5,
    batch_size: int = 0,
    seed: int = 42,
    feat_dtype: np.dtype = np.float32,
) -> list[BenchResult]:
    dtype_name = np.dtype(feat_dtype).name
    print(f"\n{'─'*60}")
    print(f"Classification  n_train={n_train:,}  n_query={n_query:,}"
          f"  d={n_features}  k={k}  classes={n_classes}  dtype={dtype_name}")
    print(f"{'─'*60}")

    train_X, train_y, query_X, query_y = _make_blobs(
        n_train, n_query, n_features, n_classes, seed, feat_dtype)

    results = []

    # --- CUDA (uses feat_dtype features) ---
    clf_cuda = KNNClassifier(k=k, batch_size=batch_size)
    clf_cuda.fit(train_X, train_y)
    t0         = time.perf_counter()
    preds_cuda = clf_cuda.predict(query_X)
    t_cuda     = time.perf_counter() - t0
    acc_cuda   = float(np.mean(preds_cuda == query_y)) * 100.0
    results.append(BenchResult(f"CUDA KNN ({dtype_name})", t_cuda, accuracy=acc_cuda))
    print(f"  CUDA:    {t_cuda:7.3f}s  accuracy={acc_cuda:.2f}%")

    # --- sklearn brute-force: always float32 (sklearn does not support fp16) ---
    train_X32 = train_X.astype(np.float32)
    query_X32 = query_X.astype(np.float32)
    clf_sk = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=-1)
    t0       = time.perf_counter()
    clf_sk.fit(train_X32, train_y)
    preds_sk = clf_sk.predict(query_X32)
    t_sk     = time.perf_counter() - t0
    acc_sk   = float(np.mean(preds_sk == query_y)) * 100.0
    results.append(BenchResult("sklearn (brute, CPU, float32)", t_sk, accuracy=acc_sk))
    print(f"  sklearn: {t_sk:7.3f}s  accuracy={acc_sk:.2f}%")

    speedup = t_sk / t_cuda if t_cuda > 0 else float("inf")
    agree   = float(np.mean(preds_cuda == preds_sk)) * 100.0
    print(f"  Speedup: {speedup:.2f}×   Agreement: {agree:.2f}%")

    return results


def bench_regression(
    n_train: int,
    n_query: int,
    n_features: int,
    k: int,
    batch_size: int = 0,
    seed: int = 42,
    feat_dtype: np.dtype = np.float32,
) -> list[BenchResult]:
    dtype_name = np.dtype(feat_dtype).name
    print(f"\n{'─'*60}")
    print(f"Regression      n_train={n_train:,}  n_query={n_query:,}"
          f"  d={n_features}  k={k}  dtype={dtype_name}")
    print(f"{'─'*60}")

    train_X, train_y, query_X, query_y = _make_regression(
        n_train, n_query, n_features, seed, feat_dtype)

    results = []

    # --- CUDA (uses feat_dtype features) ---
    reg_cuda = KNNRegressor(k=k, batch_size=batch_size)
    reg_cuda.fit(train_X, train_y)
    t0         = time.perf_counter()
    preds_cuda = reg_cuda.predict(query_X)
    t_cuda     = time.perf_counter() - t0
    mse_cuda   = float(np.mean((preds_cuda - query_y) ** 2))
    results.append(BenchResult(f"CUDA KNN ({dtype_name})", t_cuda, mse=mse_cuda))
    print(f"  CUDA:    {t_cuda:7.3f}s  MSE={mse_cuda:.6f}")

    # --- sklearn: always float32 ---
    train_X32 = train_X.astype(np.float32)
    query_X32 = query_X.astype(np.float32)
    reg_sk = KNeighborsRegressor(
        n_neighbors=k, algorithm="brute", metric="euclidean",
        weights="distance", n_jobs=-1)
    t0       = time.perf_counter()
    reg_sk.fit(train_X32, train_y)
    preds_sk = reg_sk.predict(query_X32)
    t_sk     = time.perf_counter() - t0
    mse_sk   = float(np.mean((preds_sk - query_y) ** 2))
    results.append(BenchResult("sklearn (brute, CPU, float32)", t_sk, mse=mse_sk))
    print(f"  sklearn: {t_sk:7.3f}s  MSE={mse_sk:.6f}")

    speedup = t_sk / t_cuda if t_cuda > 0 else float("inf")
    print(f"  Speedup: {speedup:.2f}×")

    return results


# ============================================================
# Dataset size matrix
# ============================================================

BENCHMARK_CONFIGS = [
    # (n_train, n_query, n_features, k)
    (   10_000,   1_000,   32,  10),
    (   50_000,   5_000,   64,  10),
    (  200_000,  10_000,   64,  15),
    (  500_000,  20_000,  128,  15),
    (1_000_000,  50_000,  128,  20),
]

QUICK_CONFIGS = [
    (10_000,  1_000, 32, 10),
    (50_000,  5_000, 64, 10),
]


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KNN-CUDA vs sklearn benchmark")
    parser.add_argument("--quick",      action="store_true",
                        help="Two small configs — fast smoke test")
    parser.add_argument("--n-train",    type=int, default=None)
    parser.add_argument("--n-query",    type=int, default=None)
    parser.add_argument("--n-features", type=int, default=None)
    parser.add_argument("--k",          type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="GPU query batch size (0 = auto-detect)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 features for the CUDA KNN path "
                             "(sklearn always uses float32)")
    args = parser.parse_args()

    feat_dtype = np.float16 if args.fp16 else np.float32

    if all(v is not None for v in [args.n_train, args.n_query, args.n_features, args.k]):
        configs = [(args.n_train, args.n_query, args.n_features, args.k)]
    elif args.quick:
        configs = QUICK_CONFIGS
    else:
        configs = BENCHMARK_CONFIGS

    print("=" * 60)
    print("  KNN-CUDA  vs  sklearn.neighbors  benchmark")
    print(f"  feature dtype : {np.dtype(feat_dtype).name}"
          + ("  (sklearn uses float32)" if args.fp16 else ""))
    print("=" * 60)

    for n_train, n_query, n_feat, k in configs:
        bench_classification(n_train, n_query, n_feat, k,
                             batch_size=args.batch_size, feat_dtype=feat_dtype)
        bench_regression(n_train, n_query, n_feat, k,
                         batch_size=args.batch_size, feat_dtype=feat_dtype)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
