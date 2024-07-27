import os
import random
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
import math
import time
from scipy.sparse.linalg import eigs
import scipy.sparse as sp
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPS = {
    # 'none': lambda C: Zero(),
    'skip_connect': lambda C: Identity(),
    'NLinear': lambda C, seq_len: Nlinear(seq_len, seq_len, C),

    'cnn': lambda C: CNN(C, C, (1, 2), dilation=1),
    'dcc_1': lambda C: DCCLayer(C, C, (1, 2), dilation=1),
    'inception': lambda C: dilated_inception_layer(C, C),

    'gru': lambda C: GRU(C, C),
    'lstm': lambda C: LSTM(C, C),

    'diff_gcn': lambda C, supports, nodevec1, nodevec2, dropout: DiffusionConvLayer(
        2, supports, nodevec1, nodevec2, C, C, dropout),
    'mix_hop': lambda C, nodevec1, nodevec2: Bidirection_mixprop(C, C, nodevec1, nodevec2),

    'trans': lambda C: InformerLayer(C, informer=False),
    'informer': lambda C: InformerLayer(C, informer=True),
    'convformer': lambda C: InformerLayer(C, informer=False, convformer=True),

    's_trans': lambda C: SpatialInformerLayer(C, informer=False),
    's_informer': lambda C: SpatialInformerLayer(C, informer=True),
    'masked_trans': lambda C, geo_mask, sem_mask: MaskedSpatialInformer(C, geo_mask, sem_mask),
}


# 从指定文件夹下读取所有参数文件并且提取arch架构，和参数
def read_params(path):
    files = os.listdir(path)
    pattern = r'\[\[.*?\]\]'  # 匹配形如 [[...]] 的字符串
    models = []
    for file_name in files:
        match = re.search(pattern, file_name)

        if match:
            result_string = match.group(0)
            result_list = eval(result_string)  # 将匹配到的字符串转换为列表
            print(result_list)
            model_param = torch.load(os.path.join(path, file_name))
            models.append((result_list, model_param))
        else:
            print("未找到匹配的字符串")
    return models


params_dir = '/home/AutoCTS++/seeds/params_pool/params'
parent_models = read_params(params_dir)

if_heavy = True

calculator = lambda tune_weight, history_weight: torch.einsum('i...,i...->...', tune_weight,
                                                              history_weight) if if_heavy else \
    torch.einsum('ji,i...->...', tune_weight, history_weight)


def shape_filter(actual_tensor, candidate_tensor_list, max_num=10):
    new_tensor_list = []
    for candidate_tensor in candidate_tensor_list:
        if actual_tensor.shape == candidate_tensor.shape:
            new_tensor_list.append(candidate_tensor)
    if max_num is not None:
        random.shuffle(new_tensor_list)

    num = min(max_num if max_num is not None else len(new_tensor_list), len(new_tensor_list))

    return new_tensor_list[0:num]


# 通过正则表达式匹配父模型的参数，并且将所有候选项返回，其中op_id代表算子编号
def search_params(op_id, key_word):
    similar_params_lst = []
    for arch, model_params in parent_models:
        # 获取第一个arch中对应算子所在的位置(第几个op)，为了后续提取参数用，由于一个结构可能有几个相同的算子，所以用一个list放index
        candidate_op_idex_list = []
        for id, (_, op) in enumerate(arch):
            if op == op_id:
                candidate_op_idex_list.append(id)
        if len(candidate_op_idex_list) == 0:
            continue

        for op_index in candidate_op_idex_list:
            pattern = r"cells\.\d+\._ops\.{}\..*".format(op_index) + key_word

            matcher = re.compile(pattern)
            similar_params = [param for name, param in model_params.items() if matcher.match(name)]
            similar_params_lst.extend(similar_params)

    if similar_params_lst:
        print(f"找到 {len(similar_params_lst)} 个与 '{key_word}' 相似的参数：")
        # for similar_param in similar_params_lst:
        #     print(similar_param[0])
    else:
        print("未找到与 '{}' 相似的参数。".format(key_word))

    return similar_params_lst


class inherit_linear(nn.Linear):
    def __init__(self, in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device: Any = None,
                 dtype: Any = None,
                 op_id: int = 0):
        super(inherit_linear, self).__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )
        self.op_id = op_id

        history_weight_list = shape_filter(actual_tensor=self.weight,
                                           candidate_tensor_list=search_params(self.op_id, 'weight'))
        history_bias_list = shape_filter(actual_tensor=self.bias,
                                         candidate_tensor_list=search_params(self.op_id, 'bias'))

        self.history_weight = torch.stack(history_weight_list)

        self.history_bias = torch.stack(history_bias_list)

        self.tune_weight = nn.Parameter(torch.rand(self.history_weight.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight.shape[0]))

        self.tune_bias = nn.Parameter(torch.rand(self.history_bias.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias.shape[0]))

        self._init_params(mode="mean")

    def _init_params(self, mode):
        if mode == 'xaiver':
            nn.init.xavier_uniform_(self.tune_weight)
            nn.init.xavier_uniform_(self.tune_bias)
        elif mode == 'mean':
            nn.init.constant_(self.tune_weight, 1 / self.history_weight.shape[0])
            nn.init.constant_(self.tune_bias, 1 / self.history_bias.shape[0])

    def forward(self, input: Tensor) -> Tensor:
        del self.weight
        del self.bias

        self.weight = calculator(self.tune_weight, self.history_weight)
        self.bias = calculator(self.tune_bias, self.history_bias)

        result = super().forward(input)

        # 欺骗torch，让其把这个动态生成的weight和bias维护在state_dict中，但是实际正向传播时重新生成
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.bias = nn.Parameter(self.bias, requires_grad=False)
        return result


class inherit_conv1d(nn.Conv1d):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple[int]],
                 stride: Union[int, tuple[int]] = 1,
                 padding: Union[str, int, tuple[int]] = 0,
                 dilation: Union[int, tuple[int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Any = None,
                 dtype: Any = None,
                 op_id: int = 0):
        super(inherit_conv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding
                                             , dilation, groups, bias, padding_mode, device, dtype)
        self.op_id = op_id

        history_weight_list = shape_filter(actual_tensor=self.weight,
                                           candidate_tensor_list=search_params(self.op_id, 'weight'))
        history_bias_list = shape_filter(actual_tensor=self.bias,
                                         candidate_tensor_list=search_params(self.op_id, 'bias'))

        self.history_weight = torch.stack(history_weight_list)

        self.history_bias = torch.stack(history_bias_list)

        self.tune_weight = nn.Parameter(torch.rand(self.history_weight.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight.shape[0]))

        self.tune_bias = nn.Parameter(torch.rand(self.history_bias.shape)) if if_heavy else nn.Parameter(
            torch.randn(1, self.history_bias.shape[0]))

        self._init_params(mode="mean")

    def _init_params(self, mode):
        if mode == 'xaiver':
            nn.init.xavier_uniform_(self.tune_weight)
            nn.init.xavier_uniform_(self.tune_bias)
        elif mode == 'mean':
            nn.init.constant_(self.tune_weight, 1 / self.history_weight.shape[0])
            nn.init.constant_(self.tune_bias, 1 / self.history_bias.shape[0])

    def forward(self, input):
        del self.weight
        del self.bias

        self.weight = calculator(self.tune_weight, self.history_weight)
        self.bias = calculator(self.tune_bias, self.history_bias)

        result = super().forward(input)

        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.bias = nn.Parameter(self.bias, requires_grad=False)

        return result


class inherit_conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, tuple[int, int]],
                 stride: Union[int, tuple[int, int]] = 1,
                 padding: Union[str, int, tuple[int, int]] = 0,
                 dilation: Union[int, tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device: Any = None,
                 dtype: Any = None,
                 op_id: int = 0
                 ):
        super(inherit_conv2d, self).__init__(in_channels,
                                             out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation,
                                             groups=groups,
                                             bias=bias,
                                             padding_mode=padding_mode,
                                             device=device,
                                             dtype=dtype)

        self.op_id = op_id

        history_weight_list = shape_filter(actual_tensor=self.weight,
                                           candidate_tensor_list=search_params(self.op_id, 'weight'))
        history_bias_list = shape_filter(actual_tensor=self.bias,
                                         candidate_tensor_list=search_params(self.op_id, 'bias'))

        self.history_weight = torch.stack(history_weight_list)

        self.history_bias = torch.stack(history_bias_list)

        self.tune_weight = nn.Parameter(torch.rand(self.history_weight.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight.shape[0]))

        self.tune_bias = nn.Parameter(torch.rand(self.history_bias.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias.shape[0]))

        self._init_params(mode="mean")

    def _init_params(self, mode):
        if mode == 'xaiver':
            nn.init.xavier_uniform_(self.tune_weight)
            nn.init.xavier_uniform_(self.tune_bias)
        elif mode == 'mean':
            nn.init.constant_(self.tune_weight, 1 / self.history_weight.shape[0])
            nn.init.constant_(self.tune_bias, 1 / self.history_bias.shape[0])

    def forward(self, input):
        del self.weight
        del self.bias

        self.weight = calculator(self.tune_weight, self.history_weight)
        self.bias = calculator(self.tune_bias, self.history_bias)

        result = super().forward(input)

        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.bias = nn.Parameter(self.bias, requires_grad=False)
        return result


class inherit_gru(nn.GRU):
    def __init__(self, c_in, c_out, bidirectional=True, batch_first=True, op_id=0):
        super(inherit_gru, self).__init__(c_in, c_out, bidirectional=bidirectional,
                                          batch_first=batch_first)
        self.op_id = op_id
        history_weight_ih_l0_list = shape_filter(actual_tensor=self.weight_ih_l0,
                                                 candidate_tensor_list=search_params(self.op_id, 'weight_ih_l0'))
        history_bias_ih_l0_list = shape_filter(actual_tensor=self.bias_ih_l0,
                                               candidate_tensor_list=search_params(self.op_id, 'bias_ih_l0'))
        history_weight_hh_l0_list = shape_filter(actual_tensor=self.weight_hh_l0,
                                                 candidate_tensor_list=search_params(self.op_id, 'weight_hh_l0'))
        history_bias_hh_l0_list = shape_filter(actual_tensor=self.bias_hh_l0,
                                               candidate_tensor_list=search_params(self.op_id, 'bias_hh_l0'))

        history_weight_ih_l0_reverse_list = shape_filter(actual_tensor=self.weight_ih_l0,
                                                         candidate_tensor_list=search_params(self.op_id,
                                                                                             'weight_ih_l0_reverse'))
        history_bias_ih_l0_reverse_list = shape_filter(actual_tensor=self.bias_ih_l0,
                                                       candidate_tensor_list=search_params(self.op_id,
                                                                                           'bias_ih_l0_reverse'))
        history_weight_hh_l0_reverse_list = shape_filter(actual_tensor=self.weight_hh_l0,
                                                         candidate_tensor_list=search_params(self.op_id,
                                                                                             'weight_hh_l0_reverse'))
        history_bias_hh_l0_reverse_list = shape_filter(actual_tensor=self.bias_hh_l0,
                                                       candidate_tensor_list=search_params(self.op_id,
                                                                                           'bias_hh_l0_reverse'))

        self.history_weight_ih_l0 = torch.stack(history_weight_ih_l0_list)
        self.history_bias_ih_l0 = torch.stack(history_bias_ih_l0_list)
        self.history_weight_hh_l0 = torch.stack(history_weight_hh_l0_list)
        self.history_bias_hh_l0 = torch.stack(history_bias_hh_l0_list)
        self.history_weight_ih_l0_reverse = torch.stack(history_weight_ih_l0_reverse_list)
        self.history_bias_ih_l0_reverse = torch.stack(history_bias_ih_l0_reverse_list)
        self.history_weight_hh_l0_reverse = torch.stack(history_weight_hh_l0_reverse_list)
        self.history_bias_hh_l0_reverse = torch.stack(history_bias_hh_l0_reverse_list)

        self.tune_weight_ih_l0 = nn.Parameter(
            torch.rand(self.history_weight_ih_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_ih_l0.shape[0]))
        self.tune_bias_ih_l0 = nn.Parameter(torch.rand(self.history_bias_ih_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_ih_l0.shape[0])
        )
        self.tune_weight_hh_l0 = nn.Parameter(
            torch.rand(self.history_weight_hh_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_hh_l0.shape[0])
        )
        self.tune_bias_hh_l0 = nn.Parameter(torch.rand(self.history_bias_hh_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_hh_l0.shape[0])
        )
        self.tune_weight_ih_l0_reverse = nn.Parameter(
            torch.rand(self.history_weight_ih_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_ih_l0_reverse.shape[0])
        )
        self.tune_bias_ih_l0_reverse = nn.Parameter(
            torch.rand(self.history_bias_ih_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_ih_l0_reverse.shape[0])

        )
        self.tune_weight_hh_l0_reverse = nn.Parameter(
            torch.rand(self.history_weight_hh_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_hh_l0_reverse.shape[0])
        )
        self.tune_bias_hh_l0_reverse = nn.Parameter(
            torch.rand(self.history_bias_hh_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_hh_l0_reverse.shape[0])
        )

        self._init_params(mode="mean")

    def _init_params(self, mode):
        if mode == 'xaiver':
            nn.init.xavier_uniform_(self.tune_weight_ih_l0)
            nn.init.xavier_uniform_(self.tune_bias_ih_l0)

            nn.init.xavier_uniform_(self.tune_weight_hh_l0)
            nn.init.xavier_uniform_(self.tune_bias_hh_l0)

            nn.init.xavier_uniform_(self.tune_weight_ih_l0_reverse)
            nn.init.xavier_uniform_(self.tune_bias_ih_l0_reverse)

            nn.init.xavier_uniform_(self.tune_weight_hh_l0_reverse)
            nn.init.xavier_uniform_(self.tune_bias_hh_l0_reverse)
        elif mode == 'mean':
            nn.init.constant_(self.tune_weight_ih_l0, 1 / self.history_weight_ih_l0.shape[0])
            nn.init.constant_(self.tune_bias_ih_l0, 1 / self.history_bias_ih_l0.shape[0])

            nn.init.constant_(self.tune_weight_hh_l0, 1 / self.history_weight_hh_l0.shape[0])
            nn.init.constant_(self.tune_bias_hh_l0, 1 / self.history_bias_hh_l0.shape[0])

            nn.init.constant_(self.tune_weight_ih_l0_reverse, 1 / self.history_weight_ih_l0_reverse.shape[0])
            nn.init.constant_(self.tune_bias_ih_l0_reverse, 1 / self.history_bias_ih_l0_reverse.shape[0])

            nn.init.constant_(self.tune_weight_hh_l0_reverse, 1 / self.history_weight_hh_l0_reverse.shape[0])
            nn.init.constant_(self.tune_bias_hh_l0_reverse, 1 / self.history_bias_hh_l0_reverse.shape[0])

    def forward(self, input, hx=None):
        del self.weight_ih_l0, self.bias_ih_l0, self.weight_hh_l0, self.bias_hh_l0, self.weight_ih_l0_reverse, self.weight_hh_l0_reverse, self.bias_hh_l0_reverse, self.bias_ih_l0_reverse

        self.weight_ih_l0 = calculator(self.tune_weight_ih_l0, self.history_weight_ih_l0)
        self.bias_ih_l0 = calculator(self.tune_bias_ih_l0, self.history_bias_ih_l0)
        self.weight_hh_l0 = calculator(self.tune_weight_hh_l0, self.history_weight_hh_l0)
        self.bias_hh_l0 = calculator(self.tune_bias_hh_l0, self.history_bias_hh_l0)

        self.weight_ih_l0_reverse = calculator(self.tune_weight_ih_l0_reverse, self.history_weight_ih_l0_reverse)

        self.bias_ih_l0_reverse = calculator(self.tune_bias_ih_l0_reverse, self.history_bias_ih_l0_reverse)

        self.weight_hh_l0_reverse = calculator(self.tune_weight_hh_l0_reverse, self.history_weight_hh_l0_reverse)

        self.bias_hh_l0_reverse = calculator(self.tune_bias_hh_l0_reverse, self.history_bias_hh_l0_reverse)

        result = super().forward(input, hx=None)

        self.weight_ih_l0 = nn.Parameter(self.weight_ih_l0, requires_grad=False)
        self.bias_ih_l0 = nn.Parameter(self.bias_ih_l0, requires_grad=False)
        self.weight_hh_l0 = nn.Parameter(self.weight_hh_l0, requires_grad=False)
        self.bias_hh_l0 = nn.Parameter(self.bias_hh_l0, requires_grad=False)
        self.weight_ih_l0_reverse = nn.Parameter(self.weight_ih_l0_reverse, requires_grad=False)
        self.bias_ih_l0_reverse = nn.Parameter(self.bias_ih_l0_reverse, requires_grad=False)
        self.weight_hh_l0_reverse = nn.Parameter(self.weight_hh_l0_reverse, requires_grad=False)
        self.bias_hh_l0_reverse = nn.Parameter(self.bias_hh_l0_reverse, requires_grad=False)

        return result


class inherit_lstm(nn.LSTM):
    def __init__(self, c_in, c_out, bidirectional=True, batch_first=True, op_id=0):
        super(inherit_lstm, self).__init__(c_in, c_out, bidirectional=bidirectional, batch_first=batch_first)
        self.op_id = op_id
        history_weight_ih_l0_list = shape_filter(actual_tensor=self.weight_ih_l0,
                                                 candidate_tensor_list=search_params(self.op_id, 'weight_ih_l0'))
        history_bias_ih_l0_list = shape_filter(actual_tensor=self.bias_ih_l0,
                                               candidate_tensor_list=search_params(self.op_id, 'bias_ih_l0'))
        history_weight_hh_l0_list = shape_filter(actual_tensor=self.weight_hh_l0,
                                                 candidate_tensor_list=search_params(self.op_id, 'weight_hh_l0'))
        history_bias_hh_l0_list = shape_filter(actual_tensor=self.bias_hh_l0,
                                               candidate_tensor_list=search_params(self.op_id, 'bias_hh_l0'))

        history_weight_ih_l0_reverse_list = shape_filter(actual_tensor=self.weight_ih_l0,
                                                         candidate_tensor_list=search_params(self.op_id,
                                                                                             'weight_ih_l0_reverse'))
        history_bias_ih_l0_reverse_list = shape_filter(actual_tensor=self.bias_ih_l0,
                                                       candidate_tensor_list=search_params(self.op_id,
                                                                                           'bias_ih_l0_reverse'))
        history_weight_hh_l0_reverse_list = shape_filter(actual_tensor=self.weight_hh_l0,
                                                         candidate_tensor_list=search_params(self.op_id,
                                                                                             'weight_hh_l0_reverse'))
        history_bias_hh_l0_reverse_list = shape_filter(actual_tensor=self.bias_hh_l0,
                                                       candidate_tensor_list=search_params(self.op_id,
                                                                                           'bias_hh_l0_reverse'))

        self.history_weight_ih_l0 = torch.stack(history_weight_ih_l0_list)
        self.history_bias_ih_l0 = torch.stack(history_bias_ih_l0_list)
        self.history_weight_hh_l0 = torch.stack(history_weight_hh_l0_list)
        self.history_bias_hh_l0 = torch.stack(history_bias_hh_l0_list)
        self.history_weight_ih_l0_reverse = torch.stack(history_weight_ih_l0_reverse_list)
        self.history_bias_ih_l0_reverse = torch.stack(history_bias_ih_l0_reverse_list)
        self.history_weight_hh_l0_reverse = torch.stack(history_weight_hh_l0_reverse_list)
        self.history_bias_hh_l0_reverse = torch.stack(history_bias_hh_l0_reverse_list)

        self.tune_weight_ih_l0 = nn.Parameter(
            torch.rand(self.history_weight_ih_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_ih_l0.shape[0]))
        self.tune_bias_ih_l0 = nn.Parameter(torch.rand(self.history_bias_ih_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_ih_l0.shape[0])
        )
        self.tune_weight_hh_l0 = nn.Parameter(
            torch.rand(self.history_weight_hh_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_hh_l0.shape[0])
        )
        self.tune_bias_hh_l0 = nn.Parameter(torch.rand(self.history_bias_hh_l0.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_hh_l0.shape[0])
        )
        self.tune_weight_ih_l0_reverse = nn.Parameter(
            torch.rand(self.history_weight_ih_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_ih_l0_reverse.shape[0])
        )
        self.tune_bias_ih_l0_reverse = nn.Parameter(
            torch.rand(self.history_bias_ih_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_ih_l0_reverse.shape[0])

        )
        self.tune_weight_hh_l0_reverse = nn.Parameter(
            torch.rand(self.history_weight_hh_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_weight_hh_l0_reverse.shape[0])
        )
        self.tune_bias_hh_l0_reverse = nn.Parameter(
            torch.rand(self.history_bias_hh_l0_reverse.shape)) if if_heavy else nn.Parameter(
            torch.rand(1, self.history_bias_hh_l0_reverse.shape[0])
        )

        self._init_params(mode='mean')

    def _init_params(self, mode):
        if mode == 'xaiver':
            nn.init.xavier_uniform_(self.tune_weight_ih_l0)
            nn.init.xavier_uniform_(self.tune_bias_ih_l0)

            nn.init.xavier_uniform_(self.tune_weight_hh_l0)
            nn.init.xavier_uniform_(self.tune_bias_hh_l0)

            nn.init.xavier_uniform_(self.tune_weight_ih_l0_reverse)
            nn.init.xavier_uniform_(self.tune_bias_ih_l0_reverse)

            nn.init.xavier_uniform_(self.tune_weight_hh_l0_reverse)
            nn.init.xavier_uniform_(self.tune_bias_hh_l0_reverse)
        elif mode == 'mean':
            nn.init.constant_(self.tune_weight_ih_l0, 1 / self.history_weight_ih_l0.shape[0])
            nn.init.constant_(self.tune_bias_ih_l0, 1 / self.history_bias_ih_l0.shape[0])

            nn.init.constant_(self.tune_weight_hh_l0, 1 / self.history_weight_hh_l0.shape[0])
            nn.init.constant_(self.tune_bias_hh_l0, 1 / self.history_bias_hh_l0.shape[0])

            nn.init.constant_(self.tune_weight_ih_l0_reverse, 1 / self.history_weight_ih_l0_reverse.shape[0])
            nn.init.constant_(self.tune_bias_ih_l0_reverse, 1 / self.history_bias_ih_l0_reverse.shape[0])

            nn.init.constant_(self.tune_weight_hh_l0_reverse, 1 / self.history_weight_hh_l0_reverse.shape[0])
            nn.init.constant_(self.tune_bias_hh_l0_reverse, 1 / self.history_bias_hh_l0_reverse.shape[0])

    def forward(self, input, hx=None):
        del self.weight_ih_l0, self.bias_ih_l0, self.weight_hh_l0, self.bias_hh_l0, self.weight_ih_l0_reverse, self.weight_hh_l0_reverse, self.bias_hh_l0_reverse, self.bias_ih_l0_reverse

        self.weight_ih_l0 = calculator(self.tune_weight_ih_l0, self.history_weight_ih_l0)
        self.bias_ih_l0 = calculator(self.tune_bias_ih_l0, self.history_bias_ih_l0)
        self.weight_hh_l0 = calculator(self.tune_weight_hh_l0, self.history_weight_hh_l0)
        self.bias_hh_l0 = calculator(self.tune_bias_hh_l0, self.history_bias_hh_l0)

        self.weight_ih_l0_reverse = calculator(self.tune_weight_ih_l0_reverse, self.history_weight_ih_l0_reverse)

        self.bias_ih_l0_reverse = calculator(self.tune_bias_ih_l0_reverse, self.history_bias_ih_l0_reverse)

        self.weight_hh_l0_reverse = calculator(self.tune_weight_hh_l0_reverse, self.history_weight_hh_l0_reverse)

        self.bias_hh_l0_reverse = calculator(self.tune_bias_hh_l0_reverse, self.history_bias_hh_l0_reverse)

        result = super().forward(input, hx=None)

        self.weight_ih_l0 = nn.Parameter(self.weight_ih_l0, requires_grad=False)
        self.bias_ih_l0 = nn.Parameter(self.bias_ih_l0, requires_grad=False)
        self.weight_hh_l0 = nn.Parameter(self.weight_hh_l0, requires_grad=False)
        self.bias_hh_l0 = nn.Parameter(self.bias_hh_l0, requires_grad=False)
        self.weight_ih_l0_reverse = nn.Parameter(self.weight_ih_l0_reverse, requires_grad=False)
        self.bias_ih_l0_reverse = nn.Parameter(self.bias_ih_l0_reverse, requires_grad=False)
        self.weight_hh_l0_reverse = nn.Parameter(self.weight_hh_l0_reverse, requires_grad=False)
        self.bias_hh_l0_reverse = nn.Parameter(self.bias_hh_l0_reverse, requires_grad=False)

        return result


######################################################################
# Nlinear
######################################################################
class Nlinear(nn.Module):
    def __init__(self, input_lengh, output_length, c_in, op_id=1):
        super(Nlinear, self).__init__()
        self.seq_len = input_lengh
        self.pred_len = output_length

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = c_in
        self.individual = False
        if self.individual:
            self.Nlinear_Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Nlinear_Linear.append(inherit_linear(self.seq_len, self.pred_len, op_id=op_id))
        else:
            self.Nlinear_Linear = inherit_linear(self.seq_len, self.pred_len, op_id=op_id)

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 170, 12, 32]
        x = x.reshape(-1, T, C)

        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Nlinear_Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Nlinear_Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last

        x = x.reshape(b, -1, T, C)
        x = x.permute(0, 3, 1, 2)

        return x  # [Batch, Output length, Channel]


######################################################################
# CNN family
######################################################################

class CausalConv2d(inherit_conv2d):
    """
    Dilated causal convolution with GTU
    单向padding，causal体现在kernel_size=2
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 op_id=0):
        self._padding = (kernel_size[-1] - 1) * dilation
        super(CausalConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=(0, self._padding),
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias, op_id=op_id)

    def forward(self, input):
        result = super(CausalConv2d, self).forward(input)
        if self._padding != 0:
            return result[:, :, :, :-self._padding]
        return result


class DCCLayer(nn.Module):
    """
    dilated causal convolution layer with GLU function
    暂时用GTU代替
    """

    def __init__(self, c_in, c_out, kernel_size, dilation=1, op_id=3):
        super(DCCLayer, self).__init__()
        # self.relu = nn.ReLU()
        # padding=0, 所以feature map尺寸会减小，最终要减小到1才行？如果没有，就要pooling
        self.dcc_filter_conv = CausalConv2d(c_in, c_out, kernel_size, dilation=dilation, op_id=op_id)
        self.dcc_gate_conv = CausalConv2d(c_in, c_out, kernel_size, dilation=dilation, op_id=op_id)
        # self.bn = nn.BatchNorm2d(c_out)
        # self.filter_conv = nn.Conv2d(c_in, c_out, kernel_size, dilation=dilation)
        # self.gate_conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        # x = self.relu(x)
        filter = self.dcc_filter_conv(x)
        gate = torch.sigmoid(self.dcc_gate_conv(x))
        output = filter * gate
        # output = self.bn(output)

        return output


class ReLUConvBN(nn.Module):
    """
    ReLu -> Conv2d -> BatchNorm2d
    """

    def __init__(self, C_in, C_out, kernel_size, stride):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))  # bias=True

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        return self.mlp(x)


class linear(nn.Module):
    """
    Linear for 2d feature map
    """

    def __init__(self, c_in, c_out, op_id=0):
        super(linear, self).__init__()
        self.mlp = inherit_conv2d(c_in, c_out, kernel_size=(1, 1), op_id=op_id)  # bias=True

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        return self.mlp(x)


class nconv(nn.Module):
    """
    张量运算
    """

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl, vw->ncwl', (x, A))
        return x.contiguous()


class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x.mul(0.)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CNN(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride=1, dilation=1, op_id=2):
        super(CNN, self).__init__()
        # self.relu = nn.ReLU()
        self.cnn_filter_conv = CausalConv2d(c_in, c_out, kernel_size, stride, dilation=dilation, op_id=op_id)
        # self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]这些block的input必须具有相同的shape？
        :return:
        """
        # x = self.relu(x)
        output = (self.cnn_filter_conv(x))
        # output = self.bn(output)

        return output


class Cheb_gcn(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out, nodevec1, nodevec2, alpha):
        """
        :param K: K-order
        :param cheb_polynomials: laplacian matrix？
        :param c_in: size of input channel
        :param c_out: size of output channel
        """
        super(Cheb_gcn, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.c_in = c_in
        self.c_out = c_out
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.alpha = alpha
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.FloatTensor(c_in, c_out).to(self.DEVICE)) for _ in range(K)])

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            self.theta_k = self.Theta[k]
            nn.init.xavier_uniform_(self.theta_k)

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch_size, N, f_in, T]

        batch_size, num_nodes, c_in, timesteps = x.shape

        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  # [b, N, f_in]
            output = torch.zeros(
                batch_size, num_nodes, self.c_out).to(self.DEVICE)  # [b, N, f_out]

            for k in range(self.K):
                alpha, beta = F.softmax(self.alpha[k])
                T_k = alpha * self.cheb_polynomials[k] + beta * adp

                # T_k = self.cheb_polynomials[k]  # [N, N]
                self.theta_k = self.Theta[k]  # [c_in, c_out]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(self.theta_k)
            outputs.append(output.unsqueeze(-1))
        outputs = F.relu(torch.cat(outputs, dim=-1)).transpose(1, 2)
        outputs = self.bn(outputs)

        return outputs


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2, op_id=4):  # dilation=2时速度很慢？？？
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(CausalConv2d(cin, cout, (1, kern), dilation=dilation_factor, op_id=op_id))
            # self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            # print(x[i].shape)  # [64, 8, 170, 12]
            x[i] = x[i][..., -x[-1].size(3):]

        x = torch.cat(x, dim=1)
        return x


class dilated_inception_layer(nn.Module):
    """from MTGNN"""

    def __init__(self, cin, cout, dilation_factor=1, op_id=4):
        super(dilated_inception_layer, self).__init__()
        self.inception_filter_conv = dilated_inception(cin, cout, dilation_factor, op_id=op_id)
        self.inception_gate_conv = dilated_inception(cin, cout, dilation_factor, op_id=op_id)

    def forward(self, x):
        filter = torch.tanh(self.inception_filter_conv(x))
        gate = torch.sigmoid(self.inception_gate_conv(x))
        output = filter * gate

        return output


######################################################################
# RNN family
######################################################################

class GRU(nn.Module):
    def __init__(self, c_in, c_out, op_id=5):
        super(GRU, self).__init__()
        # self.gru = nn.GRU(c_in, c_out, batch_first=True)
        self.gru = inherit_gru(c_in, c_out, bidirectional=True, batch_first=True, op_id=op_id)
        self.c_out = c_out

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, N, T, f_in]
        x = x.reshape(-1, T, C)  # [bN, T, f_in]
        output, state = self.gru(x)  # 双向gru会在特征维度自动拼接

        output1 = output[:, :, :self.c_out]
        output2 = output[:, :, self.c_out:]
        output = (output1 + output2)
        output = output.reshape(b, N, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


class LSTM(nn.Module):
    def __init__(self, c_in, c_out, op_id=6):
        super(LSTM, self).__init__()
        # self.lstm = nn.LSTM(c_in, c_out, batch_first=True)
        # 双向lstm比单向性能好点
        self.lstm = inherit_lstm(c_in, c_out, bidirectional=True, batch_first=True, op_id=op_id)
        self.c_out = c_out

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return:
        """
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [b, N, T, f_in]
        x = x.reshape(-1, T, C)  # [bN, T, f_in]
        output, state = self.lstm(x)
        output1 = output[:, :, :self.c_out]
        output2 = output[:, :, self.c_out:]
        output = (output1 + output2)
        output = output.reshape(b, N, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# Transformer layer
######################################################################
class LayerNorm(nn.Module):
    """
    Layer normalization.
    """

    def __init__(self, d_model, epsilon=1e-8):
        super(LayerNorm, self).__init__()
        self.epsilon = epsilon
        self.size = d_model

        self.gamma = nn.Parameter(torch.ones(self.size))
        self.beta = nn.Parameter(torch.ones(self.size))

    def forward(self, x):
        """
        :param x: [bs, T, d_model]
        :return:
        """
        normalized = (x - x.mean(dim=-1, keepdim=True)) \
                     / ((x.std(dim=-1, keepdim=True) + self.epsilon) ** .5)
        output = self.gamma * normalized + self.beta

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    """

    :param q: [bs, heads, seq_len, d_k]
    :param k:
    :param v:
    :param d_k: dim
    :param mask:
    :param dropout:
    :return:
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    seq_len = q.size(2)

    if mask is not None:  # 单向编码需要mask，取下对角阵
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)  # why

    output = torch.matmul(scores, v)

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=4, dropout=0.1):
        """

        :param d_model: input feature dimension?
        :param heads:
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads  # narrow multi-head
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        """
        :param q: [bs, T, d_model]
        :param k:
        :param v:
        :param attn_mask:
        :return: [bs, T, d_model]
        """
        batch_size = q.size(0)

        # perform linear operation and split into N heads
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)

        # transpose to get dimensions batch_size * heads * seq_len * d_k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = attention(q, k, v, self.d_k, attn_mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)

        return output


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super(Feedforward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        :param x: [bs, T, d_model]
        :return: [bs, T, d_model]
        """
        x = self.dropout(F.relu(self.linear_1(x)))
        output = self.linear_2(x)

        return output  # F.relu(output) ???


class PositionalEncoder(nn.Module):  # 都给挪到utils里面吧
    """
    add position embedding
    hyper-network包含多个transformer layer，所以需要采用固定pe
    """

    def __init__(self, d_model, max_seq_len=48, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.learnable_pe = False
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        if self.learnable_pe:
            self.pe = nn.Parameter(torch.zeros(1, max_seq_len, self.d_model))
        else:
            # create constant 'pe' matrix with values dependent on pos and i
            pe = torch.zeros(max_seq_len, d_model)
            for pos in range(max_seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / d_model)))
            pe = pe.unsqueeze(0)  # [bs, T, dim]
            self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [bs, T, dim]
        :return:
        """
        if self.learnable_pe:
            pe = Variable(self.pe[:, :x.size(1)], requires_grad=True)  # learnable position embedding
        else:
            # # make embeddings relatively larger
            # x = x * math.sqrt(self.d_model)  # why
            pe = Variable(self.pe[:, :x.size(1)], requires_grad=False)  # fixed position embedding

        if x.is_cuda:
            pe = pe.cuda()
        x = x + pe

        return self.dropout(x)


######################################################################
# Informer encoder layer
######################################################################

class SepConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1, op_id=9):
        super(SepConv1d, self).__init__()
        self.depthwise = inherit_conv1d(in_channels,
                                        in_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=in_channels, op_id=op_id)
        self.bn = nn.BatchNorm1d(in_channels)
        self.pointwise = inherit_conv1d(in_channels, out_channels, kernel_size=1, op_id=op_id)

    def forward(self, x):
        """
        :param x: [b, C, T]
        :return:
        """
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, convformer=False, op_id=9):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.convformer = convformer

        self.inner_attention = attention

        if convformer:
            kernel_size = 3
            pad = (kernel_size - 1) // 2
            self.query_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad, op_id=op_id)
            self.key_projection = SepConv1d(d_model, d_model, kernel_size, padding=pad, op_id=op_id)
            self.value_projection = SepConv1d(d_model, d_model, 1, op_id=op_id)
        else:
            self.query_projection = inherit_linear(d_model, d_keys * n_heads, op_id=op_id)
            self.key_projection = inherit_linear(d_model, d_keys * n_heads, op_id=op_id)
            self.value_projection = inherit_linear(d_model, d_values * n_heads, op_id=op_id)
        self.out_projection = inherit_linear(d_values * n_heads, d_model, op_id=op_id)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if self.convformer:
            queries = queries.transpose(-1, 1)
            keys = keys.transpose(-1, 1)
            values = values.transpose(-1, 1)

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class InformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu",
                 output_attention=False, informer=True, convformer=False, op_id=9):
        super(InformerLayer, self).__init__()
        # d_ff = d_ff or 4*d_model
        if convformer:
            op_id = 11

        if informer:
            op_id = 10
            self.attention = AttentionLayer(
                ProbAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads,
                convformer=convformer, op_id=op_id)
        else:
            self.attention = AttentionLayer(
                FullAttention(False, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads,
                convformer=convformer, op_id=op_id)

        self.conv1 = inherit_conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, op_id=op_id)
        self.conv2 = inherit_conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, op_id=op_id)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)
        self.d_model = d_model

    def forward(self, x, attn_mask=None):
        b, C, N, T = x.shape
        x = x.permute(0, 2, 3, 1)  # [64, 207, 12, 32]
        x = x.reshape(-1, T, C)  # [64*207, 12, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        # x = x * math.sqrt(self.d_model)
        # x = x + self.pe(x)
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        output = output.reshape(b, -1, T, C)
        output = output.permute(0, 3, 1, 2)

        return output


######################################################################
# GCN family
######################################################################
def scaled_Laplacian(W):
    """
    compute \tilde{L}
    :param W: adj_mx
    :return: scaled laplacian matrix
    """
    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real  # k largest real part of eigenvalues

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """
    compute a list of chebyshev polynomials from T_0 to T{K-1}
    :param L_tilde: scaled laplacian matrix
    :param K: the maximum order of chebyshev polynomials
    :return: list(np.ndarray), length: K, from T_0 to T_{K-1}
    """
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class GCNLayer(nn.Module):
    """
    K-order chebyshev graph convolution layer
    """

    def __init__(self, K, cheb_polynomials, c_in, c_out):
        """
        :param K: K-order
        :param adj_mx: original Adjacency matrix
        :param c_in: size of input channel
        :param c_out: size of output channel
        """
        super(GCNLayer, self).__init__()
        self.K = K
        self.c_in = c_in
        self.c_out = c_out
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)

        self.cheb_polynomials = cheb_polynomials
        self.DEVICE = cheb_polynomials[0].device
        self.Theta = nn.ParameterList(  # weight matrices
            [nn.Parameter(torch.randn(c_in, c_out).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        """
        :param x: [batch_size, f_in, N, T]
        :return: [batch_size, f_out, N, T]
        """
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch_size, N, f_in, T]

        batch_size, num_nodes, c_in, timesteps = x.shape

        outputs = []
        for step in range(timesteps):
            graph_signal = x[:, :, :, step]  # [b, N, f_in]
            output = torch.zeros(
                batch_size, num_nodes, self.c_out).to(self.DEVICE)  # [b, N, f_out]

            for k in range(self.K):
                T_k = self.cheb_polynomials[k]  # [N, N]
                theta_k = self.Theta[k]
                rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
                output = output + rhs.matmul(theta_k)

            outputs.append(output.unsqueeze(-1))

        outputs = torch.cat(outputs, dim=-1).transpose(1, 2)
        outputs = self.bn(outputs)
        # outputs = F.relu(torch.cat(outputs, dim=-1)).transpose(1, 2)
        # outputs = F.dropout(outputs, 0.5, training=self.training)
        return outputs


def get_normalized_adj(A):
    """
    1-order gcn
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, A_wave, c_in, c_out, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = c_in
        self.out_features = c_out
        A_wave = torch.from_numpy(A_wave)
        self.A_wave = A_wave
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        :param input: [batch_size, f_in, N, T]
        :param A_wave: Normalized adjacency matrix.
        :return:
        """
        x = input.permute(0, 2, 3, 1)  # [B, N, T, F]
        # x = self.relu(x)
        lfs = torch.einsum("ij,jklm->kilm", [self.A_wave, x.permute(1, 0, 2, 3)])
        output = F.relu(torch.matmul(lfs, self.weight))  # relu先不要吧？
        # output = (torch.matmul(lfs, self.weight))

        if self.bias is not None:
            output = output + self.bias

        output = output.permute(0, 3, 1, 2)
        # output = self.bn(output)

        return output


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


class DiffusionConvLayer(nn.Module):
    """
    Diffusion convolution layer
    K-order diffusion convolution layer with self-adaptive adjacency matrix (N, N)
    """

    def __init__(self, K, supports, nodevec1, nodevec2, c_in, c_out, dropout=False, op_id=7):
        super(DiffusionConvLayer, self).__init__()
        c_in = (K * (len(supports) + 1) + 1) * c_in
        # c_in = (K * len(supports) + 1) * c_in
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        # 之前mlp放到forward里面导致出错
        self.diffusionconv_mlp = linear(c_in, c_out, op_id=op_id)  # 7 * 32 * 32
        self.c_out = c_out
        self.K = K
        self.supports = supports
        self.nconv = nconv()
        self.dropout = dropout
        # self.relu = nn.ReLU()
        # self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        # x = self.relu(x)
        # adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # bug?
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))  # 差别不大
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)
        new_supports = self.supports + [adp]

        out = [x]
        for a in new_supports:
            # x.shape [b, dim, N, seq_len]
            # a.shape [b, N, N]
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.K + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.diffusionconv_mlp(h)  # 统一的W权重矩阵 h.shape=[64, 160, 207, 12]
        # h = self.bn(h)
        if self.dropout:
            h = F.dropout(h, 0.3, training=self.training)

        return h


class nconv_mix(nn.Module):
    def __init__(self):
        super(nconv_mix, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep=2, dropout=0., alpha=0.05, op_id=8):
        super(mixprop, self).__init__()
        self.nconv = nconv_mix()
        self.mlp = linear((gdep + 1) * c_in, c_out, op_id=op_id)
        self.gdep = gdep  # 相当于K？层数？
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class Bidirection_mixprop(nn.Module):
    """Mix-hop propagation layer in MTGNN"""

    def __init__(self, c_in, c_out, nodevec1, nodevec2, gdep=2, dropout=0., alpha=0.05, op_id=8):
        super(Bidirection_mixprop, self).__init__()
        self.mixprop1 = mixprop(c_in, c_out, gdep, dropout, alpha, op_id=8)
        self.mixprop2 = mixprop(c_in, c_out, gdep, dropout, alpha, op_id=8)
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2

    def forward(self, x):
        adp = F.relu(torch.mm(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=1)

        adj = adp
        # adj = adj + torch.eye(adj.size(0)).to(x.device)  # 这里是把自连接加上？
        output = self.mixprop1(x, adj) + self.mixprop2(x, adj.transpose(1, 0))

        return output


######################################################################
# Diffusion convolution layer with
# dynamic self-adaptive adjacency matrix in temporal dimension
######################################################################
class Diff_gcn(nn.Module):
    def __init__(self, K, supports, nodevec1, nodevec2, alpha, c_in, c_out):
        """
        diffusion gcn with self-adaptive adjacency matrix (T, N, N)
        :param K:
        :param supports:
        :param c_in:
        :param c_out:
        """
        super(Diff_gcn, self).__init__()
        self.nconv = nconv()
        c_in = (K * len(supports) + 1) * c_in
        self.nodevec1 = nodevec1
        self.nodevec2 = nodevec2
        self.mlp = linear(c_in, c_out)
        self.K = K
        self.supports = [s.unsqueeze(0).repeat(12, 1, 1) for s in supports]
        self.alpha = alpha
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(c_out, affine=False)

    def forward(self, x):
        """
        :param x: [b, C, N, T]
        :return:
        """
        x = self.relu(x)

        adp = F.relu(torch.matmul(self.nodevec1, self.nodevec2))
        mask = torch.zeros_like(adp) - 10 ** 10
        adp = torch.where(adp == 0, mask, adp)
        adp = F.softmax(adp, dim=-1)  # [T, N, N]

        # for i in range(len(self.supports)):  # 反思一下这个bug
        #     self.supports[i] = torch.FloatTensor([0.8]) * self.supports[i] + torch.FloatTensor([0.2]) * adp

        out = [x]
        x = x.permute(0, 3, 1, 2)  # (b, T, C, N)
        for i in range(len(self.supports)):
            alpha, beta = F.softmax(self.alpha[i])
            a = alpha * self.supports[i] + beta * adp
            x1 = torch.matmul(x, a)  # (b, T, C, N)(T, N, N)->
            out.append(x1.permute(0, 2, 3, 1))
            for k in range(2, self.K + 1):
                x2 = torch.matmul(x1, a)
                out.append(x2.permute(0, 2, 3, 1))
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)  # 统一的W权重矩阵 h.shape=[64, 160, 207, 12]
        h = self.bn(h)
        # h = F.dropout(h, 0.3, training=self.training)  # necessary?

        return h


######################################################################
# Spatial Transformer
######################################################################

class SpatialInformerLayer(nn.Module):
    def __init__(self, d_model, d_ff=32, dropout=0., n_heads=4, activation="relu", output_attention=False,
                 informer=True, op_id=12):
        super(SpatialInformerLayer, self).__init__()
        if informer:
            op_id = 13
            self.attention = SpatialAttentionLayer(
                SpatialProbAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads,
                op_id=op_id)
        else:
            self.attention = SpatialAttentionLayer(
                SpatialFullAttention(attention_dropout=dropout, output_attention=output_attention), d_model, n_heads,
                op_id=op_id)
        self.conv1 = inherit_conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, op_id=op_id)
        self.conv2 = inherit_conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, op_id=op_id)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # self.pe = PositionalEmbedding(d_model)  # spatial transformer需要pe吗？
        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output


class SpatialAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, op_id=12):
        super(SpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = inherit_linear(d_model, d_keys * n_heads, op_id=op_id)
        self.key_projection = inherit_linear(d_model, d_keys * n_heads, op_id=op_id)
        self.value_projection = inherit_linear(d_model, d_values * n_heads, op_id=op_id)
        self.out_projection = inherit_linear(d_values * n_heads, d_model, op_id=op_id)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # shape=[b*T, N, C]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class SpatialFullAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialFullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 在这里加上fixed邻接矩阵？
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class SpatialProbAttention(nn.Module):
    def __init__(self, factor=3, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        """
        :param Q: [b, heads, T, d_k]
        :param K: 采样的K? 长度为Ln(L_K)?
        :param sample_k: c*ln(L_k), set c=3 for now
        :param n_top: top_u queries?
        :return: Q_K and Top_k query index
        """
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        # V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)  # # [256*12, 4, 8]
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()  # [256*12, 4, 207, 8]
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) [256*12, 4, 18, 207]

        # print(context_in.shape)  # [256*12, 4, 207, 8]
        # print(torch.matmul(attn, V).shape)  # [256*12, 4, 18, 8]
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # 部分赋值
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # print(index.shape)  # [256*12, 4, 18]

        # add scale factor
        scale = self.scale or 1. / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # [256*12, 4, 18, 207] 18=sqrt(207)*3
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(2, 1).contiguous(), attn


class MaskedSpatialInformer(nn.Module):
    def __init__(self, d_model, geo_mask, sem_mask, d_ff=32, dropout=0., n_heads=8, activation="relu",
                 output_attention=False, op_id=14):
        super(MaskedSpatialInformer, self).__init__()
        self.attention = MaskedSpatialAttentionLayer(
            MaskedSpatialFullAttention(geo_mask, sem_mask, attention_dropout=dropout,
                                       output_attention=output_attention), d_model, n_heads, op_id=op_id)
        self.conv1 = inherit_conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, op_id=op_id)
        self.conv2 = inherit_conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, op_id=op_id)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.d_model = d_model

    def forward(self, x):
        b, C, N, T = x.shape
        x = x.permute(0, 3, 2, 1)  # [64, 12, 207, 32]
        x = x.reshape(-1, N, C)  # [64*12, 207, 32]
        new_x, attn = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        output = output.reshape(b, -1, N, C)
        output = output.permute(0, 3, 2, 1)

        return output


class MaskedSpatialAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, op_id=14):
        super(MaskedSpatialAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = inherit_linear(d_model, d_keys * n_heads, op_id=op_id)
        self.key_projection = inherit_linear(d_model, d_keys * n_heads, op_id=op_id)
        self.value_projection = inherit_linear(d_model, d_values * n_heads, op_id=op_id)
        self.out_projection = inherit_linear(d_values * n_heads, d_model, op_id=op_id)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        # shape=[b*T, N, C]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class MaskedSpatialFullAttention(nn.Module):
    def __init__(self, geo_mask, sem_mask, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(MaskedSpatialFullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.geo_mask = geo_mask
        self.sem_mask = sem_mask

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 两个mask
        # 要print这个scores的size，看看h是不是等于4，l和s应该都是170吧？
        # 两个head对应att1，两个head对应att2？分别使用两个mask？？？那么需要修改这里的einsum计算？应该不用
        # geo_scores = scores[:, :2, :, :]
        # sem_scores = scores[:, 2:, :, :]
        # geo_scores = geo_scores.masked_fill_(self.geo_mask, float('-inf'))  # 用-inf填充True？True有很多啊
        # sem_scores = sem_scores.masked_fill_(self.sem_mask, float('-inf'))
        # masked_scores = torch.cat([geo_scores, sem_scores], dim=1)

        # # 仅sem mask
        # scores = scores.masked_fill_(self.sem_mask, float('-inf'))
        # masked_scores = scores

        # 仅geo mask
        scores = scores.masked_fill_(self.geo_mask, float('-inf'))
        masked_scores = scores

        A = self.dropout(torch.softmax(scale * masked_scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class LaplacianPE(nn.Module):
    """
    lape_dim指的是什么维度啊，为什么是8啊，因为保留了8个最小特征值？
    lap_mx指的是函数_cal_lape()生成的拉普拉斯特征向量吧？
    这里把8维的向量转成hidden_dim维，然后170个节点每一个都对应一个32维的embedding
    """

    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


def _calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num


def cal_lape(adj_mx, lape_dim=8):
    L, isolated_point_num = _calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]).float()
    laplacian_pe.require_grad = False
    return laplacian_pe
