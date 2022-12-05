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
import copy

import torch
import numpy as np
from torch.nn import Conv2d

from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.pytorch.mixed_precision.mixed_precision_wrapper import PytorchMixedPrecisionWrapper
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.keras_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest


class base_model(torch.nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.conv1 = Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inp):
        x = self.conv1(inp)
        return x


def test_setup(representative_data_gen):
    model = base_model()
    graph = prepare_graph_with_quantization_parameters(model, PytorchImplementation(), DEFAULT_PYTORCH_INFO,
                                                       representative_data_gen, generate_pytorch_tpc,
                                                       input_shape=(1, 3, 8, 8),
                                                       mixed_precision_enabled=True)

    node = graph.get_topo_sorted_nodes()[1]

    return node


class TestPytorchSetLayerToBitwidth(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[1, 3, 8, 8]]

    def representative_data_gen(self, n_iters=1):
        input_shapes = self.create_inputs_shape()
        for _ in range(n_iters):
            yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        node = test_setup(self.representative_data_gen)
        wrapper_layer = PytorchMixedPrecisionWrapper(node, DEFAULT_PYTORCH_INFO)

        prev_attr_weights_dict = {}
        for attr in wrapper_layer.weight_attrs:
            active_weights = copy.deepcopy(wrapper_layer.layer.state_dict()[attr])
            print(active_weights)
            self.unit_test.assertTrue(active_weights is not None)
            prev_attr_weights_dict[attr] = active_weights

        wrapper_layer.set_active_weights(bitwidth_idx=1)

        for attr in wrapper_layer.weight_attrs:
            active_weights = wrapper_layer.layer.state_dict()[attr]
            self.unit_test.assertFalse(np.all(active_weights.detach().cpu().numpy() ==
                                              prev_attr_weights_dict[attr].detach().cpu().numpy()))

        wrapper_layer.set_active_weights(bitwidth_idx=0)
        for attr in wrapper_layer.weight_attrs:
            active_weights = wrapper_layer.layer.state_dict()[attr]
            self.unit_test.assertTrue(np.all(active_weights.detach().cpu().numpy() ==
                                             prev_attr_weights_dict[attr].detach().cpu().numpy()))


class TestPytorchSetSingleAttrToBitwidth(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[1, 3, 8, 8]]

    def representative_data_gen(self, n_iters=1):
        input_shapes = self.create_inputs_shape()
        for _ in range(n_iters):
            yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        node = test_setup(self.representative_data_gen)
        wrapper_layer = PytorchMixedPrecisionWrapper(node, DEFAULT_PYTORCH_INFO)

        attr = KERNEL

        active_weights = copy.deepcopy(wrapper_layer.layer.state_dict()[attr])
        self.unit_test.assertTrue(active_weights is not None)
        prev_attr_weights_dict = active_weights

        wrapper_layer.set_active_weights(bitwidth_idx=1, attr=attr)

        active_weights = wrapper_layer.layer.state_dict()[attr]
        self.unit_test.assertFalse(np.all(active_weights.detach().cpu().numpy() ==
                                          prev_attr_weights_dict.detach().cpu().numpy()))

        wrapper_layer.set_active_weights(bitwidth_idx=0)

        active_weights = wrapper_layer.layer.state_dict()[attr]
        self.unit_test.assertTrue(np.all(active_weights.detach().cpu().numpy() ==
                                         prev_attr_weights_dict.detach().cpu().numpy()))

        for attr in wrapper_layer.weight_attrs:
            active_weights = wrapper_layer.layer.state_dict()[attr]
            self.unit_test.assertTrue(np.all(active_weights.detach().cpu().numpy() ==
                                             prev_attr_weights_dict.detach().cpu().numpy()))
