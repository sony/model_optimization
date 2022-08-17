# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import model_compression_toolkit as mct
import operator

import torch
from torch import add, flatten, reshape, split, unsqueeze, dropout, sigmoid, tanh
from torch.nn import Conv2d, ConvTranspose2d, Linear, BatchNorm2d
from torch.nn import Dropout, Flatten, Hardtanh
from torch.nn import ReLU, ReLU6, PReLU, SiLU, Sigmoid, Tanh
from torch.nn.functional import relu, relu6, prelu, silu, hardtanh

from model_compression_toolkit.core.common.target_platform.targetplatform2framework import LayerFilterParams
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit.core.pytorch.reader.node_holders import DummyPlaceHolder

tp = mct.target_platform


def generate_activation_mp_tpc_pytorch(tp_model, name="activation_mp_pytorch_tp"):
    ftp_torch = tp.TargetPlatformCapabilities(tp_model,
                                              name=name)
    with ftp_torch:
        tp.OperationsSetToLayers("NoQuantization", [Dropout,
                                                     Flatten,
                                                     dropout,
                                                     flatten,
                                                     split,
                                                     operator.getitem,
                                                     reshape,
                                                     unsqueeze,
                                                     BatchNorm2d,
                                                     torch.Tensor.size])

        tp.OperationsSetToLayers("Weights_n_Activation", [Conv2d,
                                                           Linear,
                                                           ConvTranspose2d])

        tp.OperationsSetToLayers("Activation", [torch.relu,
                                                 ReLU,
                                                 ReLU6,
                                                 relu,
                                                 relu6,
                                                 LayerFilterParams(Hardtanh, min_val=0),
                                                 LayerFilterParams(hardtanh, min_val=0),
                                                 operator.add,
                                                 add,
                                                 PReLU,
                                                 prelu,
                                                 SiLU,
                                                 silu,
                                                 Sigmoid,
                                                 sigmoid,
                                                 Tanh,
                                                 tanh,
                                                 DummyPlaceHolder])

    return ftp_torch


def get_pytorch_test_tpc_dict(tp_model, test_name, ftp_name):
    return {
        test_name: generate_pytorch_tpc(name=ftp_name,
                                        tp_model=tp_model),
    }


def get_mp_activation_pytorch_tpc_dict(tpc_model, test_name, tpc_name):
    return {
        test_name: generate_activation_mp_tpc_pytorch(name=tpc_name,
                                                      tp_model=tpc_model),
    }
