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
import torch
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.network_editors.node_filters import NodeNameFilter, NodeTypeFilter
from model_compression_toolkit.core.common.network_editors.actions import EditRule, \
    ChangeCandidatesWeightsQuantizationMethod, ReplaceLayer
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

tp = mct.target_platform


class Identity(torch.nn.Module):
    """
    define custom layer as a relu replacement
    """

    def __init__(self, inplace: bool = False):
        super(Identity, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def get_identity_params_from_relu(weights={}, **kwargs):
    """
    return config and weights for the new layer (no change is required)
    """
    return weights, kwargs


class TwoLayersReluNet(torch.nn.Module):
    def __init__(self):
        super(TwoLayersReluNet, self).__init__()

        self.activation1 = torch.nn.ReLU()
        self.activation2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation1(x)
        x = self.activation2(x)
        return x


class SingleLayerReplacementTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_network_editor(self):
        return [EditRule(filter=NodeNameFilter('activation1'),
                         action=ReplaceLayer(Identity, get_identity_params_from_relu))
                ]

    def create_feature_network(self, input_shape):
        return TwoLayersReluNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quantized_model = quantized_models.get('no_quantization')
        self.unit_test.assertTrue(isinstance(quantized_model.activation1, Identity))
        self.unit_test.assertTrue(isinstance(quantized_model.activation2, torch.nn.ReLU))


class ReluReplacementTest(SingleLayerReplacementTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_network_editor(self):
        return [EditRule(filter=NodeTypeFilter(torch.nn.ReLU),
                         action=ReplaceLayer(Identity, get_identity_params_from_relu))
                ]

    def create_feature_network(self, input_shape):
        return TwoLayersReluNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quantized_model = quantized_models.get('no_quantization')
        self.unit_test.assertTrue(torch.all(torch.eq(quantized_model(input_x), input_x[0])))
        self.unit_test.assertTrue(isinstance(quantized_model.activation1, Identity))
        self.unit_test.assertTrue(isinstance(quantized_model.activation2, Identity))


class AddBias(torch.nn.Module):

    def __init__(self, bias, inplace: bool = False):
        super(AddBias, self).__init__()
        self.inplace = inplace
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.bias


def get_add_bias_params_from_relu(weights={}, **kwargs):
    """
    return config and weights for the new layer (no change is required)
    """
    if kwargs.get('inplace') is True:
        bias = 0
    else:
        bias = 3

    kwargs.update({'bias': bias})
    return weights, kwargs


class ReluReplacementWithAddBiasTest(SingleLayerReplacementTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_network_editor(self):
        return [EditRule(filter=NodeTypeFilter(torch.nn.ReLU),
                         action=ReplaceLayer(AddBias, get_add_bias_params_from_relu))
                ]

    def create_feature_network(self, input_shape):
        return TwoLayersReluNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        quantized_model = quantized_models.get('no_quantization')
        self.unit_test.assertTrue(torch.mean((quantized_model(input_x) - input_x[0])) == 6)
        self.unit_test.assertTrue(isinstance(quantized_model.activation1, AddBias))
        self.unit_test.assertTrue(isinstance(quantized_model.activation2, AddBias))
