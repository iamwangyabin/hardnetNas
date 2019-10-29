sys.path.extend(['/home/wang/workspace/hardnetNas/hardnetNAS'])
sys.path.extend(['/home/wang/workspace/hardnetNas'])
import torch
from torch import nn
from hardnetNAS.supernet_functions.lookup_table_builder import LookUpTable
from hardnetNAS.supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from hardnetNAS.supernet_functions.config_for_supernet import CONFIG_SUPERNET

input_sample = torch.autograd.Variable(torch.randn((2, 1, 32, 32)).to(torch.float32)).cuda()
lookup_table = LookUpTable(calulate_latency=False)

latency_to_accumulate = torch.autograd.Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
temperature = CONFIG_SUPERNET['train_settings']['init_temperature']
model = FBNet_Stochastic_SuperNet(lookup_table).cuda()
l = model(input_sample.cuda(), temperature, latency_to_accumulate)

