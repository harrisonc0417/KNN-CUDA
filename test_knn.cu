/*
 * test_knn.cu  --  Standalone correctness test for KNN CUDA implementation
 *
 * Tests:
 *   1. Sanity check: each point's nearest neighbour in the training set is itself
 *   2. Classification on linearly separable blobs
 *   3. Regression on a noisy sin-wave
 *
 * Build & run:
 *   make test_knn
 *   ./test_knn
 */

#include "knn.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <algorithm>
#include <vector>
#include <random>
#include <numeric>

/* ---- simple ANSI colours ---- */
#define GREEN "\033[32m"
#define RED   "\033[31m"
#define RESET "\033[0m"

static void pass(const char* name) {
    printf("  " GREEN "[PASS]" RESET " %s\n", name);
}
static void fail(const char* name, const char* msg) {
    printf("  " RED   "[FAIL]" RESET " %s  -- %s\n", name, msg);
}

/* ====================================================================
 * Test 1: Self-retrieval
 *   Query == Train.  Nearest neighbour of every point should be itself
 *   (index i, distance 0).
 * ==================================================================== */

static int test_self_retrieval() {
    const int N = 512, D = 8, K = 1;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> X(N * D);
    for (auto& v : X) v = ud(rng);

    std::vector<int>   idx(N * K);
    std::vector<float> dst(N * K);

    if (knn_search(X.data(), N, X.data(), N, D, K, 0, idx.data(), dst.data()) != 0) {
        fail("self_retrieval", "knn_search returned error");
        return 1;
    }

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (idx[i] != i)     errors++;
        if (dst[i] > 1e-4f)  errors++;
    }
    if (errors) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d/%d incorrect neighbours", errors, N);
        fail("self_retrieval", buf);
        return 1;
    }
    pass("self_retrieval");
    return 0;
}

/* ====================================================================
 * Test 2: Classification on 3-class blobs
 *   Each class is a Gaussian centred at a distinct point.
 *   With k=5 and well-separated blobs accuracy should be ~100%.
 * ==================================================================== */

static int test_classification() {
    const int N_PER_CLASS = 600, N_CLASSES = 3, D = 4, K = 5;
    const int N_TRAIN = N_PER_CLASS * N_CLASSES;
    const int N_QUERY = 300;

    /* class centroids */
    float centres[N_CLASSES][4] = {
        { 5.0f,  0.0f,  0.0f,  0.0f},
        {-5.0f,  0.0f,  0.0f,  0.0f},
        { 0.0f,  5.0f,  0.0f,  0.0f},
    };

    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 0.5f);

    std::vector<float> train_X(N_TRAIN * D);
    std::vector<int>   train_y(N_TRAIN);
    std::vector<float> query_X(N_QUERY * D);
    std::vector<int>   query_y(N_QUERY);

    /* generate training data */
    for (int c = 0; c < N_CLASSES; c++) {
        for (int i = 0; i < N_PER_CLASS; i++) {
            int n = c * N_PER_CLASS + i;
            for (int d = 0; d < D; d++)
                train_X[n * D + d] = centres[c][d] + nd(rng);
            train_y[n] = c;
        }
    }

    /* generate query data (same distribution) */
    for (int i = 0; i < N_QUERY; i++) {
        int c = i % N_CLASSES;
        for (int d = 0; d < D; d++)
            query_X[i * D + d] = centres[c][d] + nd(rng);
        query_y[i] = c;
    }

    std::vector<int>   indices(N_QUERY * K);
    std::vector<float> dists(N_QUERY * K);
    std::vector<int>   preds(N_QUERY);

    if (knn_search(train_X.data(), N_TRAIN,
                   query_X.data(), N_QUERY,
                   D, K, 0, indices.data(), dists.data()) != 0) {
        fail("classification", "knn_search returned error");
        return 1;
    }
    if (knn_classify(indices.data(), train_y.data(),
                     N_QUERY, K, N_CLASSES, preds.data()) != 0) {
        fail("classification", "knn_classify returned error");
        return 1;
    }

    int correct = 0;
    for (int i = 0; i < N_QUERY; i++)
        correct += (preds[i] == query_y[i]);

    float acc = 100.0f * correct / N_QUERY;
    if (acc < 95.0f) {
        char buf[64];
        snprintf(buf, sizeof(buf), "accuracy %.1f%% < 95%%", acc);
        fail("classification", buf);
        return 1;
    }
    printf("  " GREEN "[PASS]" RESET " classification  (accuracy %.1f%%)\n", acc);
    return 0;
}

/* ====================================================================
 * Test 3: Regression on a noisy sin wave
 *   target = sin(x) + noise(0, 0.05).
 *   KNN regression should recover the signal; MSE should be small.
 * ==================================================================== */

static int test_regression() {
    const int N_TRAIN = 2000, N_QUERY = 200, K = 7;

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> ux(0.0f, 2.0f * 3.14159f);
    std::normal_distribution<float>       noise(0.0f, 0.05f);

    std::vector<float> train_X(N_TRAIN * 1);
    std::vector<float> train_y(N_TRAIN);
    std::vector<float> query_X(N_QUERY * 1);
    std::vector<float> query_y(N_QUERY);

    for (int i = 0; i < N_TRAIN; i++) {
        float x    = ux(rng);
        train_X[i] = x;
        train_y[i] = sinf(x) + noise(rng);
    }
    for (int i = 0; i < N_QUERY; i++) {
        float x    = ux(rng);
        query_X[i] = x;
        query_y[i] = sinf(x);
    }

    std::vector<int>   indices(N_QUERY * K);
    std::vector<float> dists(N_QUERY * K);
    std::vector<float> preds(N_QUERY);

    if (knn_search(train_X.data(), N_TRAIN,
                   query_X.data(), N_QUERY,
                   1, K, 0, indices.data(), dists.data()) != 0) {
        fail("regression", "knn_search returned error");
        return 1;
    }
    if (knn_regress(indices.data(), dists.data(),
                    train_y.data(), N_QUERY, K, preds.data()) != 0) {
        fail("regression", "knn_regress returned error");
        return 1;
    }

    float mse = 0.0f;
    for (int i = 0; i < N_QUERY; i++) {
        float diff = preds[i] - query_y[i];
        mse += diff * diff;
    }
    mse /= N_QUERY;

    if (mse > 0.02f) {
        char buf[64];
        snprintf(buf, sizeof(buf), "MSE %.5f > 0.02", mse);
        fail("regression", buf);
        return 1;
    }
    printf("  " GREEN "[PASS]" RESET " regression      (MSE %.5f)\n", mse);
    return 0;
}

/* ====================================================================
 * Test 4: Ordering — distances from knn_search must be non-decreasing
 * ==================================================================== */

static int test_ordering() {
    const int N = 1000, Q = 200, D = 16, K = 10;
    std::mt19937 rng(0xDEAD);
    std::uniform_real_distribution<float> ud;

    std::vector<float> X(N * D), Xq(Q * D);
    for (auto& v : X)  v = ud(rng);
    for (auto& v : Xq) v = ud(rng);

    std::vector<int>   idx(Q * K);
    std::vector<float> dst(Q * K);

    if (knn_search(X.data(), N, Xq.data(), Q, D, K, 0, idx.data(), dst.data()) != 0) {
        fail("ordering", "knn_search returned error");
        return 1;
    }

    int errors = 0;
    for (int q = 0; q < Q; q++)
        for (int i = 1; i < K; i++)
            if (dst[q * K + i] < dst[q * K + i - 1] - 1e-5f)
                errors++;

    if (errors) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d ordering violations", errors);
        fail("ordering", buf);
        return 1;
    }
    pass("ordering");
    return 0;
}

/* ==================================================================== */

int main() {
    printf("=== KNN-CUDA correctness tests ===\n\n");

    int failures = 0;
    failures += test_self_retrieval();
    failures += test_classification();
    failures += test_regression();
    failures += test_ordering();

    printf("\n%s\n",
        failures == 0
        ? GREEN "All tests passed." RESET
        : RED   "Some tests FAILED." RESET);

    return failures ? 1 : 0;
}
