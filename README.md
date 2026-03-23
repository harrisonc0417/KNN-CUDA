# KNN-CUDA

K-Nearest Neighbors on CUDA for classification and regression.

- **Parallel (GPU):** pairwise Euclidean distance computation via a shared-memory tiled kernel
- **Sequential (CPU):** top-k selection per query using `nth_element` + `sort`, parallelised across queries with OpenMP

Tested on NVIDIA GB10 (DGX Spark, compute capability 12.1). Works on any CUDA-capable GPU.

---

## Requirements

| Dependency | Version |
|---|---|
| CUDA Toolkit | 12.x |
| g++ | 11+ |
| OpenMP | included with g++ |
| Python | 3.10+ (benchmarking only) |
| numpy, scikit-learn | (benchmarking only) |

---

## Build

```bash
make          # builds libknn.so and test_knn
make test     # builds and runs correctness tests
make clean
```

For debugging (disables most optimisations):
```bash
# Uncomment in Makefile: NVCCFLAGS += -G -g
make clean && make
```

By default `-arch=native` is used, so nvcc detects your GPU automatically. To target a specific architecture, replace it in the Makefile, e.g. `-arch=sm_90` for H100.

---

## Python setup

```bash
pip install numpy scikit-learn
```

---

## Running benchmarks

`benchmark.py` generates synthetic datasets and compares the CUDA implementation
against sklearn's brute-force CPU KNN. It imports the `KNNClassifier` and
`KNNRegressor` classes from `knn.py`.

### Quick smoke test

```bash
python3 benchmark.py --quick
```

Runs two small configs (10K and 50K training points) and prints accuracy,
MSE, and speedup for both classification and regression.

### Specific dataset size

```bash
python3 benchmark.py \
    --n-train 500000 \
    --n-query 20000  \
    --n-features 128 \
    --k 15
```

All four flags must be provided together to run a single custom config.

### Full sweep (small → 1M training points)

```bash
python3 benchmark.py
```

Runs the built-in config matrix:

| n_train   | n_query | features | k  |
|-----------|---------|----------|----|
| 10,000    | 1,000   | 32       | 10 |
| 50,000    | 5,000   | 64       | 10 |
| 200,000   | 10,000  | 64       | 15 |
| 500,000   | 20,000  | 128      | 15 |
| 1,000,000 | 50,000  | 128      | 20 |

### All flags

```
--quick             Two small configs — fast smoke test
--n-train INT       Number of training points
--n-query INT       Number of query points
--n-features INT    Feature dimensionality
--k INT             Number of neighbours
--batch-size INT    GPU query batch size (default: auto-detected)
```

---

## Using with your own dataset

`knn.py` exposes `KNNClassifier` and `KNNRegressor` with a sklearn-style
`fit` / `predict` interface. Import from it directly — `benchmark.py` is
only for benchmarking.

### Classification

```python
import numpy as np
from knn import KNNClassifier

train_X = np.load("train_features.npy").astype(np.float32)  # [n_train, n_features]
train_y = np.load("train_labels.npy").astype(np.int32)       # [n_train], labels in [0, n_classes)
query_X = np.load("query_features.npy").astype(np.float32)   # [n_query, n_features]

clf = KNNClassifier(k=10)
clf.fit(train_X, train_y)
preds = clf.predict(query_X)   # int32 [n_query]
```

### Regression

```python
import numpy as np
from knn import KNNRegressor

train_X = np.load("train_features.npy").astype(np.float32)
train_y = np.load("train_targets.npy").astype(np.float32)    # [n_train], continuous values
query_X = np.load("query_features.npy").astype(np.float32)

reg = KNNRegressor(k=10)
reg.fit(train_X, train_y)
preds = reg.predict(query_X)   # float32 [n_query]
```

### Raw neighbour indices and distances

Both classes expose `kneighbors()` if you need more than just predictions:

```python
indices, distances = clf.kneighbors(query_X)
# indices:   int32   [n_query, k]  — row indices into the training set
# distances: float32 [n_query, k]  — Euclidean distances, ascending
```

### Controlling GPU memory

Pass `batch_size` to override auto-detection:

```python
clf = KNNClassifier(k=10, batch_size=1024)
```

### C / C++

```c
#include "knn.h"

// Step 1: find neighbours
int   indices[N_QUERY * K];
float dists[N_QUERY * K];

knn_search(train_X, n_train,
           query_X, n_query,
           n_features, k,
           0,          /* batch_size: 0 = auto */
           indices, dists);

// Step 2a: classification (majority vote)
int predictions[N_QUERY];
knn_classify(indices, train_labels, n_query, k, n_classes, predictions);

// Step 2b: regression (inverse-distance-weighted average)
float predictions[N_QUERY];
knn_regress(indices, dists, train_targets, n_query, k, predictions);
```

Link against `libknn.so`:
```bash
gcc -o myapp myapp.c -L. -lknn -Wl,-rpath,.
```

---

## Memory and batch sizing

The distance matrix is computed in batches of query points to fit in GPU memory. The default auto-sizing uses 25% of free GPU memory per batch, with a hard cap to prevent 32-bit integer overflow in the kernel (`q_idx * n_train` must not exceed `INT_MAX`).

Rough memory usage per batch:
```
GPU:  batch_size × n_train × 4 bytes   (distance matrix)
      n_train × n_features × 4 bytes   (training data, loaded once)
Host: batch_size × n_train × 4 bytes   (pinned buffer for DtH copy)
```

For `n_train=1M`, `batch_size=2000`: ~8 GB GPU + ~8 GB pinned host.

---

## File overview

```
knn.h            C API declaration (knn_search, knn_classify, knn_regress)
knn.cu           CUDA distance kernel + CPU top-k + C implementations
knn.py           Python KNNClassifier / KNNRegressor classes (import this)
benchmark.py     Timing + accuracy comparison vs sklearn.neighbors
test_knn.cu      Standalone C++ correctness tests
Makefile
requirements.txt
CUDA_EXPLAINED.md   Detailed walkthrough of the CUDA kernel
```
