#ifndef MOE_GROUPED_GEMM_HOST_HPP
#define MOE_GROUPED_GEMM_HOST_HPP

#include "helper.hpp"

void moegroupedgemm_gelu_forward(Tensor batch, Tensor a, Tensor b, Tensor c);
void moegroupedgemm_forward(Tensor batch, Tensor a, Tensor b, Tensor c);

void moegroupedgemm_gelu_backward(Tensor batch, Tensor grad_out, Tensor a, Tensor b, Tensor c_out);
void moegroupedgemm_backward(Tensor batch, Tensor grad_out, Tensor a, Tensor b);

#endif // MOE_GROUPED_GEMM_HOST_HPP