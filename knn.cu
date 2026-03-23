/*
 * knn.cu  --  K-Nearest Neighbors: CUDA distance kernel + CPU top-k
 *
 * ┌─────────────────────────────────────────────────────────────────────┐
 * │  PARALLEL  (GPU)  euclidean_distance_kernel                        │
 * │    One thread per (query_point, train_point) pair.                 │
 * │    Shared-memory tiling over the feature dimension reduces         │
 * │    global-memory bandwidth by a factor of TILE.                    │
 * │                                                                     │
 * │  SEQUENTIAL (CPU) select_top_k                                     │
 * │    Per query point: std::nth_element  O(n_train)                   │
 * │                   + std::sort of k elems  O(k log k)               │
 * └─────────────────────────────────────────────────────────────────────┘
 */

#include "knn.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <omp.h>

/* ====================================================================
 * Error-checking helpers
 * ==================================================================== */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            return -1;                                                      \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_GOTO(call, lbl)                                          \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            rc = -1; goto lbl;                                              \
        }                                                                   \
    } while (0)

/* ====================================================================
 * PARALLEL: GPU Distance Kernel
 *
 * Grid:  (ceil(n_train / TILE), ceil(n_query / TILE))
 * Block: (TILE, TILE)
 *
 * Thread layout inside a block:
 *   threadIdx.x (tx) → dimension along n_train
 *   threadIdx.y (ty) → dimension along n_query
 *
 * Shared tiles:
 *   Qs[ty][tx] = query [q_idx][feature f+tx]   – coalesced row load
 *   Ts[tx][ty] = train [t_idx][feature f+ty]   – transposed so the
 *                                                  accumulation reads
 *                                                  Ts[tx][i] contiguously
 *
 * After all feature tiles: dist[q_idx][t_idx] = sqrt(sum of sq diffs)
 * ==================================================================== */

#define TILE 16

__global__ void euclidean_distance_kernel(
    const float* __restrict__ query,   /* [n_query × n_features] */
    const float* __restrict__ train,   /* [n_train × n_features] */
    float*       __restrict__ dist,    /* [n_query × n_train]    */
    int n_query, int n_train, int n_features
) {
    __shared__ float Qs[TILE][TILE];   /* query sub-block               */
    __shared__ float Ts[TILE][TILE];   /* train sub-block (transposed)  */

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int q_idx = blockIdx.y * TILE + ty;
    const int t_idx = blockIdx.x * TILE + tx;

    float accum = 0.0f;

    const int n_feat_tiles = (n_features + TILE - 1) / TILE;

    for (int ft = 0; ft < n_feat_tiles; ft++) {
        const int f = ft * TILE;

        /* Coalesced load: thread (tx,ty) reads query[q_idx, f+tx] */
        Qs[ty][tx] = (q_idx < n_query && f + tx < n_features)
                   ? query[q_idx * n_features + f + tx]
                   : 0.0f;

        /* Transposed load: thread (tx,ty) reads train[t_idx, f+ty]
         * stored at Ts[tx][ty] so that Ts[tx][i] = train[t_idx, f+i] */
        Ts[tx][ty] = (t_idx < n_train && f + ty < n_features)
                   ? train[t_idx * n_features + f + ty]
                   : 0.0f;

        __syncthreads();

        /* Accumulate squared differences over this tile */
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            float diff = Qs[ty][i] - Ts[tx][i];
            accum += diff * diff;
        }

        __syncthreads();
    }

    if (q_idx < n_query && t_idx < n_train)
        dist[(size_t)q_idx * n_train + t_idx] = sqrtf(accum);
}

/* ====================================================================
 * SEQUENTIAL: CPU Top-K Selection
 *
 * For each query row in dist[0..batch_q):
 *   1. std::nth_element  → O(n_train)   puts k smallest at front
 *   2. std::sort of [0,k) → O(k log k)  orders them by distance
 *
 * This is intentionally left on the CPU to contrast with the GPU
 * distance computation above.
 *
 * OpenMP parallelises across query points; each thread owns its own
 * idx scratch buffer to avoid data races.
 * ==================================================================== */

static void select_top_k(
    const float* dist,     /* [batch_q × n_train]  distances   */
    int*         indices,  /* [batch_q × k]        output idx  */
    float*       out_dists,/* [batch_q × k]        output dist (nullable) */
    int batch_q, int n_train, int k
) {
    #pragma omp parallel
    {
        std::vector<int> idx(n_train);   /* one scratch buf per thread */

        #pragma omp for schedule(dynamic, 16)
        for (int q = 0; q < batch_q; q++) {
            const float* dq = dist + (size_t)q * n_train;
            int*         iq = indices + q * k;

            std::iota(idx.begin(), idx.end(), 0);

            /* O(n_train): place the k smallest indices at the front */
            std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                [dq](int a, int b) { return dq[a] < dq[b]; });

            /* O(k log k): sort those k to ascending distance order */
            std::sort(idx.begin(), idx.begin() + k,
                [dq](int a, int b) { return dq[a] < dq[b]; });

            for (int i = 0; i < k; i++)
                iq[i] = idx[i];

            if (out_dists) {
                float* od = out_dists + q * k;
                for (int i = 0; i < k; i++)
                    od[i] = dq[idx[i]];
            }
        }
    }
}

/* ====================================================================
 * Automatic batch size: use 70 % of free GPU memory for the distance
 * matrix, capped at 8 192 query points per batch.
 * ==================================================================== */

static int auto_batch_size(int n_train) {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    /* Use 25% of free memory (need matching pinned host buf + GPU buf) */
    size_t usable      = (size_t)(free_mem * 0.25);
    size_t bytes_per_q = (size_t)n_train * sizeof(float);
    int batch = (bytes_per_q > 0) ? (int)(usable / bytes_per_q) : 1;
    /* Hard cap to prevent int32 overflow in kernel: q_idx * n_train <= INT_MAX */
    int safe_cap = (n_train > 0)
        ? (int)((size_t)2147483647u / (size_t)n_train) - 1
        : 4096;
    if (batch < 1)                            batch = 1;
    if (batch > 4096)                         batch = 4096;
    if (safe_cap > 0 && batch > safe_cap)     batch = safe_cap;
    return batch;
}

/* ====================================================================
 * knn_search  (exported)
 * ==================================================================== */

int knn_search(
    const float* train_X, int n_train,
    const float* query_X, int n_query,
    int n_features, int k, int batch_size,
    int*   out_indices,
    float* out_dists
) {
    if (batch_size <= 0)
        batch_size = auto_batch_size(n_train);

    int rc = 0;

    /* ---- Allocations ---- */
    float* d_train = nullptr;
    float* d_query = nullptr;
    float* d_dist  = nullptr;
    float* h_dist  = nullptr;   /* pinned host buffer for fast DtH copies */

    CUDA_CHECK_GOTO(cudaMalloc(&d_train,
        (size_t)n_train * n_features * sizeof(float)), cleanup);

    CUDA_CHECK_GOTO(cudaMemcpy(d_train, train_X,
        (size_t)n_train * n_features * sizeof(float),
        cudaMemcpyHostToDevice), cleanup);

    CUDA_CHECK_GOTO(cudaMalloc(&d_query,
        (size_t)batch_size * n_features * sizeof(float)), cleanup);

    CUDA_CHECK_GOTO(cudaMalloc(&d_dist,
        (size_t)batch_size * n_train * sizeof(float)), cleanup);

    /* Pinned host buffer: DMA-able, avoids staging through a bounce buffer */
    CUDA_CHECK_GOTO(cudaMallocHost(&h_dist,
        (size_t)batch_size * n_train * sizeof(float)), cleanup);

    {
        dim3 block(TILE, TILE);
        int processed = 0;

        while (processed < n_query) {
            int bq = std::min(batch_size, n_query - processed);

            /* ---- Copy query batch to device ---- */
            CUDA_CHECK_GOTO(cudaMemcpy(d_query,
                query_X + (size_t)processed * n_features,
                (size_t)bq * n_features * sizeof(float),
                cudaMemcpyHostToDevice), cleanup);

            /* ---- PARALLEL: launch distance kernel ---- */
            dim3 grid(
                (n_train + TILE - 1) / TILE,
                (bq      + TILE - 1) / TILE
            );
            euclidean_distance_kernel<<<grid, block>>>(
                d_query, d_train, d_dist, bq, n_train, n_features);

            CUDA_CHECK_GOTO(cudaGetLastError(),      cleanup);
            CUDA_CHECK_GOTO(cudaDeviceSynchronize(), cleanup);

            /* ---- Copy distances back to pinned host memory ---- */
            CUDA_CHECK_GOTO(cudaMemcpy(h_dist, d_dist,
                (size_t)bq * n_train * sizeof(float),
                cudaMemcpyDeviceToHost), cleanup);

            /* ---- SEQUENTIAL: top-k on CPU (OpenMP parallel over queries) ---- */
            int*   qi = out_indices + (size_t)processed * k;
            float* qd = out_dists   ? out_dists + (size_t)processed * k
                                    : nullptr;
            select_top_k(h_dist, qi, qd, bq, n_train, k);

            processed += bq;
        }
    }

cleanup:
    if (d_train) cudaFree(d_train);
    if (d_query) cudaFree(d_query);
    if (d_dist)  cudaFree(d_dist);
    if (h_dist)  cudaFreeHost(h_dist);
    return rc;
}

/* ====================================================================
 * knn_classify  (exported)
 *
 * SEQUENTIAL: majority vote over k neighbours.  Ties broken by the
 * class with the smallest index.
 * ==================================================================== */

int knn_classify(
    const int* indices,
    const int* train_labels,
    int n_query, int k, int n_classes,
    int* predictions
) {
    std::vector<int> votes(n_classes);

    for (int q = 0; q < n_query; q++) {
        const int* knn = indices + (size_t)q * k;
        std::fill(votes.begin(), votes.end(), 0);
        for (int i = 0; i < k; i++)
            votes[train_labels[knn[i]]]++;
        predictions[q] = (int)(
            std::max_element(votes.begin(), votes.end()) - votes.begin());
    }
    return 0;
}

/* ====================================================================
 * knn_regress  (exported)
 *
 * SEQUENTIAL: inverse-distance-weighted average over k neighbours.
 * weight_i = 1 / (dist_i + ε)  to avoid divide-by-zero.
 * ==================================================================== */

int knn_regress(
    const int*   indices,
    const float* dists,
    const float* train_targets,
    int n_query, int k,
    float* predictions
) {
    for (int q = 0; q < n_query; q++) {
        const int*   knn  = indices + (size_t)q * k;
        const float* kdst = dists   + (size_t)q * k;

        float wsum = 0.0f, wtot = 0.0f;
        for (int i = 0; i < k; i++) {
            float w  = 1.0f / (kdst[i] + 1e-8f);
            wsum    += w * train_targets[knn[i]];
            wtot    += w;
        }
        predictions[q] = (wtot > 0.0f) ? wsum / wtot : 0.0f;
    }
    return 0;
}
