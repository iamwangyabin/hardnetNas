import torch
import torch.nn as nn
import torch.nn.functional as F

class NormConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(NormConv, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.feature(x)
        return out

xx = NormConv(16, 16, 1)



import torch
from torch import nn
from hardnetNAS.supernet_functions.lookup_table_builder import LookUpTable
from hardnetNAS.supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from hardnetNAS.supernet_functions.config_for_supernet import CONFIG_SUPERNET

input_sample = torch.autograd.Variable(torch.randn((10, 3, 32, 32)).to(torch.float32)).cuda()
lookup_table = LookUpTable(calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])
latency_to_accumulate = torch.autograd.Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
temperature = CONFIG_SUPERNET['train_settings']['init_temperature']
model = FBNet_Stochastic_SuperNet(lookup_table).cuda()
model(input_sample.cuda(), temperature, latency_to_accumulate)

