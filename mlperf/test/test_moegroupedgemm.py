import torch
from torch import nn
import mlperf


class MoEGroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, expert_num=1, act_gelu=False, dropout_rate=0.0, assign=None):
        super(MoEGroupedLinear, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.expert_num   = expert_num
        self.weight = nn.Parameter(torch.empty(expert_num, out_features, in_features, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(expert_num, out_features, dtype=torch.bfloat16))
        self.act_gelu = act_gelu
        self.dropout_rate = dropout_rate

        if assign:
            self.weight.data.fill_(assign[0])
            self.bias.data.fill_(assign[1])
        else:
            nn.init.normal_(self.weight, mean=0, std=1)
            nn.init.normal_(self.bias, mean=0, std=1)

    def forward(self, input_tensor, expert_token_cnt):
        return mlperf.ops.moEGroupedLinear(expert_token_cnt, input_tensor, self.weight, self.bias, self.act_gelu, self.dropout_rate)

    def extra_repr(self):
        return 'in_features={}, out_features={}, expert_num={},'.format(self.in_features, self.out_features, self.expert_num)

if __name__ == "__main__":
    in_dim = 1024
    expert_num = 5
    
    expert_token_cnt = torch.tensor([i+2 for i in range(expert_num)], 
                                   dtype=torch.int64, 
                                   device="cuda:0")
    
    test1 = MoEGroupedLinear(in_dim, in_dim, expert_num).cuda()
    
    input_list = [
        torch.randn( (expert_token_cnt[i].item(), in_dim),
                    device="cuda:0", dtype=torch.bfloat16)
        for i in range(len(expert_token_cnt))
    ]
    
    print("Input shapes:", [x.shape for x in input_list])
    input_ = torch.cat(input_list, dim=0)

    # 执行前向传播
    output = test1(input_, expert_token_cnt)
    print("Output shape:", output.shape)