# sys.path.extend(['/home/wang/workspace/hardnetNas/hardnetNAS'])
# sys.path.extend(['/home/wang/workspace/hardnetNas'])
# import torch
# from torch import nn
# from hardnetNAS.supernet_functions.lookup_table_builder import LookUpTable
# from hardnetNAS.supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
# from hardnetNAS.supernet_functions.config_for_supernet import CONFIG_SUPERNET
#
# lookup_table = LookUpTable(calulate_latency=False)
#
# input_sample = torch.autograd.Variable(torch.randn((2, 1, 32, 32)).to(torch.float32)).cuda()
# latency_to_accumulate = torch.autograd.Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
# temperature = CONFIG_SUPERNET['train_settings']['init_temperature']
# model = FBNet_Stochastic_SuperNet(lookup_table).cuda()
# l = model(input_sample.cuda(), temperature, latency_to_accumulate)
#
#
# from hardnetNAS.fbnet_building_blocks.fbnet_builder import IRFBlock
# import torch
# import torch.nn as nn
# import time
#
# class HardNetNeiMask(nn.Module):
#     def __init__(self):
#         super(HardNetNeiMask, self).__init__()
#         x = lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
#             C_in, C_out, 1, stride, kernel=3, **kwargs)
#         self.features = nn.Sequential(
#             x(32,32,-999,2)
#         )
#     def forward(self, input):
#         input = self.features(input)
#         return input
#
# input_sample = torch.randn(1000, 32, 32, 32).to(torch.float32).cuda()
#
# model = HardNetNeiMask().cuda()
#
# l=0
# for i in range(10):
#     # input_sample = torch.randn(1000, 32, 32, 32).to(torch.float32).cuda()
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     model(input_sample)
#     end.record()
#     torch.cuda.synchronize()
#     l += start.elapsed_time(end)
# print(l/100)


# sys.path.extend(['/home/wang/workspace/hardnetNas/hardnetNAS'])
# sys.path.extend(['/home/wang/workspace/hardnetNas'])
import eval
from general_functions.utils import get_logger, weights_init, load, create_directories_from_list, \
    check_tensor_in_list, writh_new_ARCH_to_fbnet_modeldef
from supernet_functions.model_supernet import FBNet_Stochastic_SuperNet, SupernetLoss
from supernet_functions.lookup_table_builder import LookUpTable
from supernet_functions.config_for_supernet import CONFIG_SUPERNET
from torch import nn


lookup_table = LookUpTable(calulate_latency=CONFIG_SUPERNET['lookup_table']['create_from_scratch'])
model = FBNet_Stochastic_SuperNet(lookup_table).cuda()
model = nn.DataParallel(model)
load(model, "/home/wang/workspace/hardnetNas/hardnetNAS/supernet_functions/logs/score1.042_acc0.613_lat22.330.pth")
eval.evaluation(model, CONFIG_SUPERNET)








#
#
#
#
# for i in range(10):
#     time0 = time.time()
#     model(input_sample)
#     time1 = time.time()
#     torch.cuda.synchronize()
#     l += time1-time0
#
#
# from hardnetNAS.fbnet_building_blocks.fbnet_builder import IRFBlock
# import torch
# import torch.nn as nn
# import time
#
# class HardNetNeiMask(nn.Module):
#     def __init__(self):
#         super(HardNetNeiMask, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
#             IRFBlock(32, 32, 0.5, 2, kernel=3),
#             IRFBlock(32, 64, 0.5, 2, kernel=3),
#             IRFBlock(64, 128, 0.5, 1, kernel=3),
#             IRFBlock(128, 128, 0.5, 2, kernel=3),
#             nn.Conv2d(128, 128, kernel_size=4, bias=False),
#             nn.BatchNorm2d(128, affine=False),
#         )
#
#     @staticmethod
#     def input_norm(x):
#         eps = 1e-8
#         flat = x.view(x.size(0), -1)
#         mp = torch.mean(flat, dim=1)
#         sp = torch.std(flat, dim=1) + eps
#         return (
#             x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
#         ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
#
#     def forward(self, input):
#         x_features = self.features(self.input_norm(input))
#         x = x_features.view(x_features.size(0), -1)
#         feature = x / torch.norm(x, p=2, dim=-1, keepdim=True)
#         return feature
#
# model = HardNetNeiMask().cuda()
# input_sample = torch.randn((1000, 1, 32, 32)).to(torch.float32).cuda()
# model(input_sample)
# for i in range(100):
#     model(input_sample)
#
# total_time=0
# for i in range(100):
#     input_sample = torch.randn((1000, 1, 32, 32)).to(torch.float32).cuda()
#     torch.cuda.synchronize()
#     time0 = time.time()
#     model(input_sample)
#     torch.cuda.synchronize()
#     time1 = time.time()
#     total_time += (time1 - time0)
#
# print(total_time/100)