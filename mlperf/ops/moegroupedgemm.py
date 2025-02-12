import importlib
from typing import Tuple
import torch
from torch.autograd import Function
from mlperf.ext import moegroupedgemm_forward, moegroupedgemm_backward, moegroupedgemm_gelu_forward, moegroupedgemm_gelu_backward


class MoEGroupedLinearFunction(Function):
    @staticmethod
    def forward(ctx, expert_token_cnt, input_tensor, weight, bias, act_gelu = False, dropout_rate = 0.0):
        if act_gelu:
            output, bias_out = moegroupedgemm_gelu_forward(expert_token_cnt, input_tensor, weight, bias)
        else:
            output = moegroupedgemm_forward(expert_token_cnt, input_tensor, weight, bias)
            bias_out = torch.Tensor(0)

        ctx.save_for_backward(expert_token_cnt, input_tensor, weight, bias_out)
        ctx.act_gelu = act_gelu
        ctx.dropout_rate = dropout_rate
        return output

    @staticmethod
    def backward(ctx, grad_out):
        expert_token_cnt, input_tensor, weight, bias_out = ctx.saved_tensors
        if ctx.act_gelu:
            grad_in, grad_weight, grad_bias = moegroupedgemm_gelu_backward(expert_token_cnt, grad_out, input_tensor, weight, bias_out)
        else:
            grad_in, grad_weight, grad_bias = moegroupedgemm_backward(expert_token_cnt, grad_out, input_tensor, weight)
        return None, grad_in, grad_weight, grad_bias, None, None


moEGroupedLinear = MoEGroupedLinearFunction.apply