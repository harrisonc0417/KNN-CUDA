"""
knn.py  --  Python wrapper around libknn.so

Exposes two classes with a sklearn-compatible interface:

    KNNClassifier  -- majority-vote classification
    KNNRegressor   -- inverse-distance-weighted regression

Both share the same underlying CUDA distance kernel and CPU top-k
selection via the C library built with `make libknn.so`.

Usage
-----
    from knn import KNNClassifier, KNNRegressor

    clf = KNNClassifier(k=10)
    clf.fit(train_X, train_y)
    preds = clf.predict(query_X)

    reg = KNNRegressor(k=10)
    reg.fit(train_X, train_y)
    preds = reg.predict(query_X)

    # Raw neighbour indices and distances
    indices, distances = clf.kneighbors(query_X)   # shape [n_query, k]
"""

import ctypes
import os

import numpy as np

# ============================================================
# Load shared library
# ============================================================

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libknn.so")

try:
    _lib = ctypes.CDLL(_LIB_PATH)
except OSError as e:
    raise RuntimeError(
        f"Could not load {_LIB_PATH}.\n"
        f"Run `make libknn.so` first.\n"
        f"Original error: {e}"
    )

# ---- knn_search ----
# int knn_search(train_X, n_train, query_X, n_query,
#                n_features, k, batch_size, out_indices, out_dists)
_lib.knn_search.restype  = ctypes.c_int
_lib.knn_search.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # train_X
    ctypes.c_int,                    # n_train
    ctypes.POINTER(ctypes.c_float),  # query_X
    ctypes.c_int,                    # n_query
    ctypes.c_int,                    # n_features
    ctypes.c_int,                    # k
    ctypes.c_int,                    # batch_size (0 = auto)
    ctypes.POINTER(ctypes.c_int),    # out_indices
    ctypes.POINTER(ctypes.c_float),  # out_dists
]

# ---- knn_classify ----
# int knn_classify(indices, train_labels, n_query, k, n_classes, predictions)
_lib.knn_classify.restype  = ctypes.c_int
_lib.knn_classify.argtypes = [
    ctypes.POINTER(ctypes.c_int),    # indices
    ctypes.POINTER(ctypes.c_int),    # train_labels
    ctypes.c_int,                    # n_query
    ctypes.c_int,                    # k
    ctypes.c_int,                    # n_classes
    ctypes.POINTER(ctypes.c_int),    # predictions
]

# ---- knn_regress ----
# int knn_regress(indices, dists, train_targets, n_query, k, predictions)
_lib.knn_regress.restype  = ctypes.c_int
_lib.knn_regress.argtypes = [
    ctypes.POINTER(ctypes.c_int),    # indices
    ctypes.POINTER(ctypes.c_float),  # dists
    ctypes.POINTER(ctypes.c_float),  # train_targets
    ctypes.c_int,                    # n_query
    ctypes.c_int,                    # k
    ctypes.POINTER(ctypes.c_float),  # predictions
]

# ============================================================
# Internal helpers
# ============================================================

def _fptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    assert arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def _iptr(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_int):
    assert arr.dtype == np.int32 and arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def _contiguous_f32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)

def _contiguous_i32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


# ============================================================
# Base class
# ============================================================

class _KNNBase:
    """
    Shared fit / kneighbors logic for both classifier and regressor.

    Parameters
    ----------
    k          : number of neighbours
    batch_size : query batch size passed to the CUDA kernel.
                 0 (default) lets the library auto-detect based on free GPU memory.
    """

    def __init__(self, k: int, batch_size: int = 0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k          = k
        self.batch_size = batch_size
        self._train_X   = None   # float32 [n_train, n_features]
        self._train_y   = None   # set by subclass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_KNNBase":
        """
        Store training data.  No computation happens here — all work
        is deferred to predict() / kneighbors().

        Parameters
        ----------
        X : array-like, shape [n_train, n_features]
        y : array-like, shape [n_train]
        """
        self._train_X = _contiguous_f32(np.asarray(X))
        self._store_labels(y)
        return self

    def _store_labels(self, y: np.ndarray):
        raise NotImplementedError

    def _check_fitted(self):
        if self._train_X is None:
            raise RuntimeError("Call fit() before predict() or kneighbors().")

    def kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the k nearest neighbours for each row of X.

        Parameters
        ----------
        X : array-like, shape [n_query, n_features]

        Returns
        -------
        indices   : int32 array [n_query, k]  — row indices into the training set
        distances : float32 array [n_query, k] — Euclidean distances, ascending
        """
        self._check_fitted()
        query_X = _contiguous_f32(np.asarray(X))

        n_train, n_feat = self._train_X.shape
        n_query         = query_X.shape[0]

        if query_X.ndim != 2 or query_X.shape[1] != n_feat:
            raise ValueError(
                f"X must have shape [n_query, {n_feat}], got {query_X.shape}"
            )

        indices   = np.empty((n_query, self.k), dtype=np.int32)
        distances = np.empty((n_query, self.k), dtype=np.float32)

        rc = _lib.knn_search(
            _fptr(self._train_X), n_train,
            _fptr(query_X),       n_query,
            n_feat, self.k, self.batch_size,
            _iptr(indices),
            _fptr(distances),
        )
        if rc != 0:
            raise RuntimeError("knn_search failed — check stderr for CUDA errors")

        return indices, distances


# ============================================================
# Classifier
# ============================================================

class KNNClassifier(_KNNBase):
    """
    K-Nearest Neighbours classifier backed by a CUDA distance kernel.

    Parallel   (GPU):  pairwise Euclidean distances via euclidean_distance_kernel
    Sequential (CPU):  top-k selection, then majority vote

    Parameters
    ----------
    k          : number of neighbours
    batch_size : GPU query batch size (0 = auto)

    Example
    -------
        clf = KNNClassifier(k=10)
        clf.fit(train_X, train_y)          # train_y: integer labels [0, n_classes)
        preds = clf.predict(query_X)       # int32 array [n_query]
        indices, dists = clf.kneighbors(query_X)
    """

    def _store_labels(self, y: np.ndarray):
        self._train_y  = _contiguous_i32(np.asarray(y))
        self.n_classes = int(self._train_y.max()) + 1
        self.classes_  = np.arange(self.n_classes, dtype=np.int32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each row of X via majority vote.

        Parameters
        ----------
        X : array-like, shape [n_query, n_features]

        Returns
        -------
        predictions : int32 array [n_query]
        """
        indices, _ = self.kneighbors(X)
        n_query    = indices.shape[0]
        preds      = np.empty(n_query, dtype=np.int32)

        rc = _lib.knn_classify(
            _iptr(indices),
            _iptr(self._train_y),
            n_query, self.k, self.n_classes,
            _iptr(preds),
        )
        if rc != 0:
            raise RuntimeError("knn_classify failed")
        return preds


# ============================================================
# Regressor
# ============================================================

class KNNRegressor(_KNNBase):
    """
    K-Nearest Neighbours regressor backed by a CUDA distance kernel.

    Parallel   (GPU):  pairwise Euclidean distances via euclidean_distance_kernel
    Sequential (CPU):  top-k selection, then inverse-distance-weighted average

    Parameters
    ----------
    k          : number of neighbours
    batch_size : GPU query batch size (0 = auto)

    Example
    -------
        reg = KNNRegressor(k=10)
        reg.fit(train_X, train_y)          # train_y: continuous float values
        preds = reg.predict(query_X)       # float32 array [n_query]
        indices, dists = reg.kneighbors(query_X)
    """

    def _store_labels(self, y: np.ndarray):
        self._train_y = _contiguous_f32(np.asarray(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for each row of X via inverse-distance weighting.

        Parameters
        ----------
        X : array-like, shape [n_query, n_features]

        Returns
        -------
        predictions : float32 array [n_query]
        """
        indices, distances = self.kneighbors(X)
        n_query            = indices.shape[0]
        preds              = np.empty(n_query, dtype=np.float32)

        rc = _lib.knn_regress(
            _iptr(indices),
            _fptr(distances),
            _fptr(self._train_y),
            n_query, self.k,
            _fptr(preds),
        )
        if rc != 0:
            raise RuntimeError("knn_regress failed")
        return preds
