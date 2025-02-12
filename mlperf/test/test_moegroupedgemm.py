import torch
from torch import nn
import mlperf

class MoEGroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, expert_num = 1, act_gelu = False, dropout_rate = 0.0, assign=None):
        super(MoEGroupedLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.expert_num   = expert_num
        self.weight = nn.Parameter(torch.Tensor(expert_num, out_features, in_features))
        self.bias   = nn.Parameter(torch.Tensor(expert_num, out_features))
        self.act_gelu     = act_gelu
        self.dropout_rate = dropout_rate

        if assign:
            self.weight.data.fill_(assign[0])
            self.bias.data.fill_(assign[1])
        else:
            torch.nn.init.normal_(self.weight, mean=0, std=1)
            torch.nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input_tensor, expert_token_cnt):
        return mlperf.moEGroupedLinear(expert_token_cnt, input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate)

    def extra_repr(self):
        return 'in_features={}, out_features={}, expert_num={},'.format(self.in_features, self.out_features, self.expert_num)

