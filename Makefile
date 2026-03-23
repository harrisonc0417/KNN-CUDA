# ============================================================
# KNN-CUDA  --  Makefile
#
# Targets:
#   libknn.so    shared library (loaded by Python via ctypes)
#   test_knn     standalone correctness test binary
#   test         build + run test_knn
#   clean        remove build artefacts
#
# GPU: NVIDIA GB10 (DGX Spark), compute capability 12.1
# -arch=native  lets nvcc detect the exact SM version automatically.
# ============================================================

NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17 -arch=native \
             --compiler-options=-fPIC,-O3,-fopenmp \
             --generate-line-info
LDFLAGS   := -fopenmp

# For debugging: add -G -g (disables most optimisations)
# NVCCFLAGS += -G -g

LIB_SRCS  := knn.cu
LIB_OUT   := libknn.so

BIN_SRCS  := test_knn.cu knn.cu
BIN_OUT   := test_knn

.PHONY: all test clean

all: $(LIB_OUT) $(BIN_OUT)

$(LIB_OUT): $(LIB_SRCS) knn.h
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $(LIB_SRCS)
	@echo "Built $@"

$(BIN_OUT): $(BIN_SRCS) knn.h
	$(NVCC) $(NVCCFLAGS) -o $@ $(BIN_SRCS) --compiler-options=$(LDFLAGS)
	@echo "Built $@"

test: $(BIN_OUT)
	./$(BIN_OUT)

clean:
	rm -f $(LIB_OUT) $(BIN_OUT)
