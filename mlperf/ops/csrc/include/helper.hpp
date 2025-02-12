#ifndef HELPER_HPP
#define HELPER_HPP

#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "cutlass/cutlass.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

#define DIVUP(m, n) (((m) + (n) - 1) / (n))

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

using at::Half;
using at::Tensor;
using phalf = at::Half;

#endif // HELPER_HPP