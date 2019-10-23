# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import (
    BatchNorm2d,
    Conv2d,
    FrozenBatchNorm2d,
    interpolate,
    _NewEmptyTensorOp
)
from .fbnet_modeldef import MODEL_ARCH

logger = logging.getLogger(__name__)

def _py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _get_divisible_by(num, divisible_by, min_val):
    ret = int(num)
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((_py2_round(num / divisible_by) or min_val) * divisible_by)
    return ret

# InvertedResidual
# inp, oup, stride, benchmodel
# MobileBlock
# in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size

PRIMITIVES = {
    # layer search 1 stride=2
    "ShuffleV2_A_1": lambda C_in, C_out, stride, benchmodel,**kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "ShuffleV2_B_1": lambda C_in, C_out, stride, benchmodel,**kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "MobileV3_RE_1": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "MobileV3_Hswish_1": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "conv_k3_1": lambda C_in, C_out, stride, **kwargs: NormConv(
        C_in, C_out, stride
    ),
    "MaxPooling": lambda C_in, C_out, **kwargs: DownMaxPool(),

    # layer search 2 stride=1
    "skip_2": lambda C_in, C_out, **kwargs: Skip(),
    "ShuffleV2_A_2": lambda C_in, C_out, stride, benchmodel,**kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "ShuffleV2_B_2": lambda C_in, C_out, stride, benchmodel, **kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "MobileV3_RE_2": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "MobileV3_Hswish_2": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "conv_k3_2": lambda C_in, C_out, stride, **kwargs: NormConv(
        C_in, C_out, stride
    ),

    # layer search 3 stride=2
    "ShuffleV2_A_3": lambda C_in, C_out, stride, benchmodel,**kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "ShuffleV2_B_3": lambda C_in, C_out, stride, benchmodel,**kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "MobileV3_RE_3": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "MobileV3_Hswish_3": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "conv_k3_3": lambda C_in, C_out, stride, **kwargs: NormConv(
        C_in, C_out, stride
    ),

    # layer search 4 stride=1
    "skip_4": lambda C_in, C_out, **kwargs: Skip(),
    "ShuffleV2_A_4": lambda C_in, C_out, stride, benchmodel, **kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "ShuffleV2_B_4": lambda C_in, C_out, stride, benchmodel, **kwargs: InvertedResidual(
        C_in, C_out, stride, benchmodel
    ),
    "MobileV3_RE_4": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "MobileV3_Hswish_4": lambda C_in, C_out, stride, nonLinear, SE, exp_size, **kwargs: MobileBlock(
        C_in, C_out, 3, stride, nonLinear, SE, exp_size
    ),
    "conv_k3_4": lambda C_in, C_out, stride, **kwargs: NormConv(
        C_in, C_out, stride
    ),

    # layer search 5 stride=1,stride=2
    "ShuffleV2_A_5": lambda C_in, C_out, expansion, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "ShuffleV2_B_5": lambda C_in, C_out, expansion, stride, **kwargs: Identity(
        C_in, C_out, stride
    ),
    "MobileV3_RE_5": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, **kwargs
    ),
    "MobileV3_Hswish_5": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, **kwargs
    ),
    "conv_k3_5_stride1": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=5, **kwargs
    ),
    "conv_k3_5_stride2": lambda C_in, C_out, expansion, stride, **kwargs: IRFBlock(
        C_in, C_out, expansion, stride, kernel=5, **kwargs
    ),
}

class Skip(nn.Module):
    def __init__(self):
        super(Skip, self).__init__()

    def forward(self, x):
        out = x
        return out

class NormConv(nn.Module):
    def __init__(self, inp, oup, stride):
        super(NormConv, self).__init__()
        self.feature = conv_bn(inp, oup, stride)

    def forward(self, x):
        out = self.feature(x)
        return out

class DownMaxPool(nn.Module):
    def __init__(self):
        super(DownMaxPool, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.MaxPool(x)
        return out

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        else:
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x

class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        # MobileNetV2
        out = self.conv(x)
        out = self.depth_conv(out)

        # Squeeze and Excite
        if self.SE:
            out = self.squeeze_block(out)

        # point-wise conv
        out = self.point_conv(out)

        # connection
        if self.use_connect:
            return x + out
        else:
            return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class ConvBNRelu(nn.Sequential):
    def __init__(
            self,
            input_depth,
            output_depth,
            kernel,
            stride,
            pad,
            no_bias,
            use_relu,
            bn_type,
            group=1,
            *args,
            **kwargs
    ):
        super(ConvBNRelu, self).__init__()

        assert use_relu in ["relu", None]
        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]
        assert bn_type in ["bn", "af", "gn", None]
        assert stride in [1, 2, 4]

        op = Conv2d(
            input_depth,
            output_depth,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            bias=not no_bias,
            groups=group,
            *args,
            **kwargs
        )
        nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
        if op.bias is not None:
            nn.init.constant_(op.bias, 0.0)
        self.add_module("conv", op)

        if bn_type == "bn":
            bn_op = BatchNorm2d(output_depth)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=output_depth)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(output_depth)
        if bn_type is not None:
            self.add_module("bn", bn_op)

        if use_relu == "relu":
            self.add_module("relu", nn.ReLU(inplace=True))


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=None):
        super(Upsample, self).__init__()
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return interpolate(
            x, scale_factor=self.scale, mode=self.mode,
            align_corners=self.align_corners
        )


def _get_upsample_op(stride):
    assert (
            stride in [1, 2, 4]
            or stride in [-1, -2, -4]
            or (isinstance(stride, tuple) and all(x in [-1, -2, -4] for x in stride))
    )

    scales = stride
    ret = None
    if isinstance(stride, tuple) or stride < 0:
        scales = [-x for x in stride] if isinstance(stride, tuple) else -stride
        stride = 1
        ret = Upsample(scale_factor=scales, mode="nearest", align_corners=None)

    return ret, stride




def _expand_block_cfg(block_cfg):
    assert isinstance(block_cfg, list)
    ret = []
    for idx in range(block_cfg[2]):
        cur = copy.deepcopy(block_cfg)
        cur[2] = 1
        cur[3] = 1 if idx >= 1 else cur[3]
        ret.append(cur)
    return ret


def expand_stage_cfg(stage_cfg):
    """ For a single stage """
    assert isinstance(stage_cfg, list)
    ret = []
    for x in stage_cfg:
        ret += _expand_block_cfg(x)
    return ret


def expand_stages_cfg(stage_cfgs):
    """ For a list of stages """
    assert isinstance(stage_cfgs, list)
    ret = []
    for x in stage_cfgs:
        ret.append(expand_stage_cfg(x))
    return ret


def _block_cfgs_to_list(block_cfgs):
    assert isinstance(block_cfgs, list)
    ret = []
    for stage_idx, stage in enumerate(block_cfgs):
        stage = expand_stage_cfg(stage)
        for block_idx, block in enumerate(stage):
            cur = {"stage_idx": stage_idx, "block_idx": block_idx, "block": block}
            ret.append(cur)
    return ret


def _add_to_arch(arch, info, name):
    """ arch = [{block_0}, {block_1}, ...]
        info = [
            # stage 0
            [
                block0_info,
                block1_info,
                ...
            ], ...
        ]
        convert to:
        arch = [
            {
                block_0,
                name: block0_info,
            },
            {
                block_1,
                name: block1_info,
            }, ...
        ]
    """
    assert isinstance(arch, list) and all(isinstance(x, dict) for x in arch)
    assert isinstance(info, list) and all(isinstance(x, list) for x in info)
    idx = 0
    for stage_idx, stage in enumerate(info):
        for block_idx, block in enumerate(stage):
            assert (
                    arch[idx]["stage_idx"] == stage_idx
                    and arch[idx]["block_idx"] == block_idx
            ), "Index ({}, {}) does not match for block {}".format(
                stage_idx, block_idx, arch[idx]
            )
            assert name not in arch[idx]
            arch[idx][name] = block
            idx += 1


def unify_arch_def(arch_def):
    """ unify the arch_def to:
        {
            ...,
            "arch": [
                {
                    "stage_idx": idx,
                    "block_idx": idx,
                    ...
                },
                {}, ...
            ]
        }
    """
    ret = copy.deepcopy(arch_def)

    assert "block_cfg" in arch_def and "stages" in arch_def["block_cfg"]
    assert "stages" not in ret
    # copy 'first', 'last' etc. inside arch_def['block_cfg'] to ret
    ret.update({x: arch_def["block_cfg"][x] for x in arch_def["block_cfg"]})
    ret["stages"] = _block_cfgs_to_list(arch_def["block_cfg"]["stages"])
    del ret["block_cfg"]

    assert "block_op_type" in arch_def
    _add_to_arch(ret["stages"], arch_def["block_op_type"], "block_op_type")
    del ret["block_op_type"]

    return ret


def get_num_stages(arch_def):
    ret = 0
    for x in arch_def["stages"]:
        ret = max(x["stage_idx"], ret)
    ret = ret + 1
    return ret


def get_blocks(arch_def, stage_indices=None, block_indices=None):
    ret = copy.deepcopy(arch_def)
    ret["stages"] = []
    for block in arch_def["stages"]:
        keep = True
        if stage_indices not in (None, []) and block["stage_idx"] not in stage_indices:
            keep = False
        if block_indices not in (None, []) and block["block_idx"] not in block_indices:
            keep = False
        if keep:
            ret["stages"].append(block)
    return ret


class FBNetBuilder(object):
    def __init__(
            self,
            width_ratio,
            bn_type="bn",
            width_divisor=1,
            dw_skip_bn=False,
            dw_skip_relu=False,
    ):
        self.width_ratio = width_ratio
        self.last_depth = -1
        self.bn_type = bn_type
        self.width_divisor = width_divisor
        self.dw_skip_bn = dw_skip_bn
        self.dw_skip_relu = dw_skip_relu

    def add_first(self, stage_info, dim_in=3, pad=True):
        # stage_info: [c, s, kernel]
        assert len(stage_info) >= 2
        channel = stage_info[0]
        stride = stage_info[1]
        out_depth = self._get_divisible_width(int(channel * self.width_ratio))
        kernel = 3
        if len(stage_info) > 2:
            kernel = stage_info[2]

        out = ConvBNRelu(
            dim_in,
            out_depth,
            kernel=kernel,
            stride=stride,
            pad=kernel // 2 if pad else 0,
            no_bias=1,
            use_relu="relu",
            bn_type=self.bn_type,
        )
        self.last_depth = out_depth
        return out

    def add_blocks(self, blocks):
        """ blocks: [{}, {}, ...]
        """
        assert isinstance(blocks, list) and all(
            isinstance(x, dict) for x in blocks
        ), blocks

        modules = OrderedDict()
        for block in blocks:
            stage_idx = block["stage_idx"]
            # print("stage_idx =", stage_idx)

            block_idx = block["block_idx"]
            block_op_type = block["block_op_type"]
            tcns = block["block"]
            n = tcns[2]
            assert n == 1
            nnblock = self.add_ir_block(tcns, [block_op_type])
            nn_name = "xif{}_{}".format(stage_idx, block_idx)
            assert nn_name not in modules
            modules[nn_name] = nnblock
        ret = nn.Sequential(modules)
        return ret

    # def add_final_pool(self, model, blob_in, kernel_size):
    #     ret = model.AveragePool(blob_in, "final_avg", kernel=kernel_size, stride=1)
    #     return ret

    def _add_ir_block(
            self, dim_in, dim_out, stride, expand_ratio, block_op_type, **kwargs
    ):
        ret = PRIMITIVES[block_op_type](
            dim_in,
            dim_out,
            expansion=expand_ratio,
            stride=stride,
            bn_type=self.bn_type,
            width_divisor=self.width_divisor,
            dw_skip_bn=self.dw_skip_bn,
            dw_skip_relu=self.dw_skip_relu,
            **kwargs
        )
        return ret, ret.output_depth

    def add_ir_block(self, tcns, block_op_types, **kwargs):
        t, c, n, s = tcns
        assert n == 1
        out_depth = self._get_divisible_width(int(c * self.width_ratio))
        dim_in = self.last_depth
        op, ret_depth = self._add_ir_block(
            dim_in,
            out_depth,
            stride=s,
            expand_ratio=t,
            block_op_type=block_op_types[0],
            **kwargs
        )
        self.last_depth = ret_depth
        return op

    def _get_divisible_width(self, width):
        ret = _get_divisible_by(int(width), self.width_divisor, self.width_divisor)
        return ret

    def add_last_states(self, cnt_classes, dropout_ratio=0.2):
        assert cnt_classes >= 1
        op = nn.Sequential(OrderedDict([
            ("conv_k1", nn.Conv2d(self.last_depth, 1504, kernel_size=1)),
            ("dropout", nn.Dropout(dropout_ratio)),
            ("avg_pool_k7", nn.AvgPool2d(kernel_size=7)),
            ("flatten", Flatten()),
            ("fc", nn.Linear(in_features=1504, out_features=cnt_classes)),
        ]))
        self.last_depth = cnt_classes
        return op


def _get_trunk_cfg(arch_def):
    num_stages = get_num_stages(arch_def)
    trunk_stages = arch_def.get("backbone", range(num_stages - 1))
    ret = get_blocks(arch_def, stage_indices=trunk_stages)
    return ret


class FBNet(nn.Module):
    def __init__(
            self, builder, arch_def, dim_in, cnt_classes=1000
    ):
        super(FBNet, self).__init__()
        self.first = builder.add_first(arch_def["first"], dim_in=dim_in)
        trunk_cfg = _get_trunk_cfg(arch_def)
        self.stages = builder.add_blocks(trunk_cfg["stages"])
        self.last_stages = builder.add_last_states(cnt_classes)

    def forward(self, x):
        y = self.first(x)
        y = self.stages(y)
        y = self.last_stages(y)
        return y


def get_model(arch, cnt_classes):
    assert arch in MODEL_ARCH
    arch_def = MODEL_ARCH[arch]
    arch_def = unify_arch_def(arch_def)
    builder = FBNetBuilder(width_ratio=1.0, bn_type="bn", width_divisor=8, dw_skip_bn=True, dw_skip_relu=True)
    model = FBNet(builder, arch_def, dim_in=3, cnt_classes=cnt_classes)
    return model
