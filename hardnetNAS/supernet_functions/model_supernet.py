import torch
from torch import nn
from collections import OrderedDict
from fbnet_building_blocks.fbnet_builder import ConvBNRelu, Flatten
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

import general_functions.Losses as Losses

class MixedOperation(nn.Module):
    # Arguments:
    # proposed_operations is a dictionary {operation_name : op_constructor}
    # latency is a dictionary {operation_name : latency}
    def __init__(self, layer_parameters, proposed_operations, latency):
        super(MixedOperation, self).__init__()
        ops_names = [op_name for op_name in proposed_operations]
        
        self.ops = nn.ModuleList([proposed_operations[op_name](*layer_parameters)
                                  for op_name in ops_names])
        self.latency = [latency[op_name] for op_name in ops_names]
        self.thetas = nn.Parameter(torch.Tensor([1.0 / len(ops_names) for i in range(len(ops_names))]))
    
    def forward(self, x, temperature, latency_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        # soft_mask_variables = nn.functional.softmax(self.thetas)
        output  = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))

        latency = sum(m * lat for m, lat in zip(soft_mask_variables, self.latency))
        latency_to_accumulate = latency_to_accumulate + latency
        # import pdb
        # pdb.set_trace()
        nmsprobs = self.softnms(soft_mask_variables.unsqueeze(0).unsqueeze(0),9,50)
        soft_sample_latency = sum(m * lat for m, lat in zip(nmsprobs.squeeze(), self.latency))
        hard_sample_latency = self.latency[torch.argmax(soft_mask_variables).item()]
        # print(str(soft_sample_latency.item())+' '+str(hard_sample_latency))
        return output, latency_to_accumulate, soft_sample_latency, hard_sample_latency

    # 一维的NMS
    def softnms(self, variables, ksize, strength):
        maxk = torch.nn.functional.max_pool1d(variables,ksize,stride=1,padding=ksize//2)
        max_all, max_all_idx = maxk.max(dim=-1, keepdim=True)
        exp_maps = torch.exp(strength * (variables - max_all))
        exp_maps_pad = torch.nn.functional.pad(exp_maps, [ksize//2, ksize//2], mode='replicate')
        sum_exp = torch.nn.functional.conv1d(exp_maps_pad, weight=torch.ones([1, 1, ksize]).to(exp_maps.device), stride=1)
        probs = exp_maps / sum_exp
        return probs

class FBNet_Stochastic_SuperNet(nn.Module):
    def __init__(self, lookup_table):
        super(FBNet_Stochastic_SuperNet, self).__init__()
        
        self.first = ConvBNRelu(input_depth=1, output_depth=32, kernel=3, stride=1,
                                pad=1, no_bias=1, use_relu="relu", bn_type="bn")
        self.stages_to_search = nn.ModuleList([MixedOperation(
                                                   lookup_table.layers_parameters[layer_id],
                                                   lookup_table.lookup_table_operations,
                                                   lookup_table.lookup_table_latency[layer_id])
                                               for layer_id in range(lookup_table.cnt_layers)])
        self.last_stages = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(lookup_table.layers_parameters[-1][1], 128, kernel_size = 4, bias=False)),
            ("batchnorm", nn.BatchNorm2d(128, affine=False)),
            ("flatten", Flatten()),
        ]))
    
    def forward(self, x, temperature, latency_to_accumulate):        
        y = self.first(x)
        soft=0
        hard=0
        for mixed_op in self.stages_to_search:
            y, latency_to_accumulate, softsample, hardsample = mixed_op(y, temperature, latency_to_accumulate)
            soft+=softsample
            hard+=hardsample
        # import pdb
        # pdb.set_trace()
        # print('latency_to_accumulate:\t'+str(latency_to_accumulate.item())+'\tsoft:\t'+str(soft.item()) + '\thard:\t' + str(hard))
        y = self.last_stages(y)
        return y, latency_to_accumulate, soft, hard
    
class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        self.weight_criterion = nn.CrossEntropyLoss()
        self.weight_criterion_hardnet = Losses.loss_HardNet

    def forward(self, outs, targets, latency, sample_latency):
        ce = self.weight_criterion_hardnet(outs, targets)

        if (sample_latency-20)>0:
            # lat = torch.log(latency ** self.beta) * (torch.log(torch.tensor(sample_latency-20+1))**0.5)
            lat = torch.log(latency ** self.beta) * ((sample_latency-20)/20)**1.07
        else:
            lat = torch.log(latency ** self.beta) * 0
        # lat = torch.log(latency ** self.beta)
        # import pdb
        # pdb.set_trace()
        # loss = self.alpha * ce * lat
        loss = self.alpha * (ce + lat)
        return loss, ce, lat

