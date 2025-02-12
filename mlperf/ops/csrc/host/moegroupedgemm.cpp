#include <iostream>
#include "helper.hpp"

void MoRGroupedLinearForwardKernelLaucher(const at::Tensor& batch, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);
void MoRGroupedLinearGeluForwardKernelLaucher(const at::Tensor& batch, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c);
void MoRGroupedLinearBackwardKernelLaucher(const at::Tensor& batch, const at::Tensor& grad_out, const at::Tensor& a, const at::Tensor& b);
void MoRGroupedLinearGeluBackwardKernelLaucher(const at::Tensor& batch, const at::Tensor& grad_out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c_out);

void moegroupedgemm_forward(at::Tensor batch, at::Tensor a, at::Tensor b, at::Tensor c)
{
    if (batch.device().type() != at::kCUDA || a.device().type() != at::kCUDA || b.device().type() != at::kCUDA || c.device().type() != at::kCUDA) {
        throw std::runtime_error("Tensor must be on CUDA device");
    } else {
        MoRGroupedLinearForwardKernelLaucher(batch, a, b, c);
    }
}

void moegroupedgemm_gelu_forward(at::Tensor batch, at::Tensor a, at::Tensor b, at::Tensor c)
{
    if (batch.device().type() != at::kCUDA || a.device().type() != at::kCUDA || b.device().type() != at::kCUDA || c.device().type() != at::kCUDA) {
        throw std::runtime_error("Tensor must be on CUDA device");
    } else {
        MoRGroupedLinearForwardKernelLaucher(batch, a, b, c);
    }
}

void moegroupedgemm_backward(at::Tensor batch, at::Tensor grad_out, at::Tensor a, at::Tensor b)
{
    if (batch.device().type() != at::kCUDA || grad_out.device().type() != at::kCUDA || a.device().type() != at::kCUDA || b.device().type() != at::kCUDA) {
        throw std::runtime_error("Tensor must be on CUDA device");
    } else {
        MoRGroupedLinearBackwardKernelLaucher(batch, grad_out, a, b);
    }
}

void moegroupedgemm_gelu_backward(at::Tensor batch, at::Tensor grad_out, at::Tensor a, at::Tensor b, at::Tensor c_out)
{
    if (batch.device().type() != at::kCUDA || grad_out.device().type() != at::kCUDA || a.device().type() != at::kCUDA || b.device().type() != at::kCUDA || c_out.device().type() != at::kCUDA) {
        throw std::runtime_error("Tensor must be on CUDA device");
    } else {
        MoRGroupedLinearGeluBackwardKernelLaucher(batch, grad_out, a, b, c_out);
    }
}
