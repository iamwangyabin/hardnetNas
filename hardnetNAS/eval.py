import torch
import torch.nn.init
import torch.nn as nn
from torch.autograd import Variable
import cv2
import time
import numpy as np
import glob
import os

descr_name = 'HardNet'

# all types of patches
tps = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5', 'h1', 'h2', 'h3', 'h4', 'h5', \
       't1', 't2', 't3', 't4', 't5']

class hpatches_sequence:
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = tps

    def __init__(self, base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t + '.png')
            im = cv2.imread(im_path, 0)
            self.N = im.shape[0] / 65
            setattr(self, t, np.split(im, self.N))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(-1).expand_as(x)
        return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim=1) + self.eps
        x = x / norm.expand_as(x)
        return x

input_dir = "../hpatches-benchmark/data/hpatches-release"
output_dir = "../hpatches-benchmark/data/descriptors"
seqs = glob.glob(input_dir + '/*')
seqs = [os.path.abspath(p) for p in seqs]

w = 65

def evaluation(model,CONFIG_SUPERNET):
    curr_desc_name = descr_name
    for seq_path in seqs:
        seq = hpatches_sequence(seq_path)
        path = os.path.join(output_dir, os.path.join(curr_desc_name, seq.name))
        if not os.path.exists(path):
            os.makedirs(path)
        descr = np.zeros((int(seq.N), 128))  # trivial (mi,sigma) descriptor
        for tp in tps:
            print(seq.name + '/' + tp)
            if os.path.isfile(os.path.join(path, tp + '.csv')):
                continue
            n_patches = 0
            for i, patch in enumerate(getattr(seq, tp)):
                n_patches += 1
            t = time.time()
            patches_for_net = np.zeros((n_patches, 1, 32, 32))
            uuu = 0
            for i, patch in enumerate(getattr(seq, tp)):
                patches_for_net[i, 0, :, :] = cv2.resize(patch[0:w, 0:w], (32, 32))
            ###
            model.eval()
            outs = []
            bs = 128;
            n_batches = n_patches / bs + 1
            # import pdb
            # pdb.set_trace()
            for batch_idx in range(int(n_batches)):
                st = batch_idx * bs
                if batch_idx == n_batches - 1:
                    if (batch_idx + 1) * bs > n_patches:
                        end = n_patches
                    else:
                        end = (batch_idx + 1) * bs
                else:
                    end = (batch_idx + 1) * bs
                if st >= end:
                    continue
                data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
                data_a = torch.from_numpy(data_a)
                data_a = data_a.cuda()
                data_a = Variable(data_a, volatile=True)
                # compute output
                latency_to_accumulate = torch.autograd.Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
                temperature = CONFIG_SUPERNET['train_settings']['init_temperature']

                out_a,_,_,_ = model(data_a,temperature,latency_to_accumulate)
                outs.append(out_a.data.cpu().numpy().reshape(-1, 128))
            res_desc = np.concatenate(outs)
            print(res_desc.shape, n_patches)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches, -1))
            np.savetxt(os.path.join(path, tp + '.csv'), out, delimiter=',', fmt='%10.5f')  # X is an array