#pragma once
/*
 * knn.h  --  K-Nearest Neighbors on CUDA
 *
 * Parallel   (GPU):  pairwise Euclidean distance computation
 * Sequential (CPU):  top-k selection, classification vote, regression average
 *
 * All host arrays are row-major, float32 (features) / int32 (labels).
 * The library is compiled as a shared object (libknn.so) so it can be
 * loaded from Python via ctypes for benchmarking.
 */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * knn_search
 * ----------
 * Find the k nearest training points for every query point.
 *
 * GPU (parallel):  euclidean_distance_kernel computes all n_query × n_train
 *                  distances in parallel using a shared-memory tiled approach.
 * CPU (sequential): select_top_k uses std::nth_element + std::sort to obtain
 *                   the ordered k-nearest neighbours for each query point.
 *
 * train_X      [n_train  × n_features]  training features (host ptr)
 * query_X      [n_query  × n_features]  query   features (host ptr)
 * n_train      number of training points
 * n_query      number of query points
 * n_features   feature dimensionality
 * k            number of neighbours
 * batch_size   query batch size for GPU memory management (0 = auto-detect)
 * out_indices  [n_query × k]  output: neighbour indices into train_X
 * out_dists    [n_query × k]  output: corresponding distances (may be NULL)
 *
 * Returns 0 on success, -1 on CUDA error.
 */
int knn_search(
    const float* train_X, int n_train,
    const float* query_X, int n_query,
    int n_features, int k, int batch_size,
    int*   out_indices,
    float* out_dists        /* nullable */
);

/*
 * knn_classify
 * ------------
 * Majority-vote classification from k nearest neighbours.
 * Sequential CPU implementation.
 *
 * indices      [n_query × k]   from knn_search
 * train_labels [n_train]       integer class labels in [0, n_classes)
 * predictions  [n_query]       output: predicted class per query point
 */
int knn_classify(
    const int* indices,
    const int* train_labels,
    int n_query, int k, int n_classes,
    int* predictions
);

/*
 * knn_regress
 * -----------
 * Inverse-distance-weighted regression from k nearest neighbours.
 * Sequential CPU implementation.
 *
 * indices        [n_query × k]  from knn_search
 * dists          [n_query × k]  from knn_search
 * train_targets  [n_train]      continuous target values
 * predictions    [n_query]      output: predicted value per query point
 */
int knn_regress(
    const int*   indices,
    const float* dists,
    const float* train_targets,
    int n_query, int k,
    float* predictions
);

#ifdef __cplusplus
}
#endif
