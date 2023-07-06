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
from mct_quantizers import PytorchQuantizationWrapper, PytorchActivationQuantizationHolder
from torch.nn import Conv2d

from model_compression_toolkit.core.common.mixed_precision.set_layer_to_bitwidth import set_layer_to_bitwidth
from model_compression_toolkit.core.pytorch.constants import KERNEL
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.pytorch.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_quantization_parameters
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

    layer = list(model.children())[0]
    node = graph.get_topo_sorted_nodes()[1]

    return node, layer


class TestSetLayerToBitwidthWeights(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[1, 3, 8, 8]]

    def representative_data_gen(self, n_iters=1):
        input_shapes = self.create_inputs_shape()
        for _ in range(n_iters):
            yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        node, layer = test_setup(self.representative_data_gen)
        wrapper_layer = PytorchQuantizationWrapper(layer,
                                                   weights_quantizers={KERNEL:
                                                       ConfigurableWeightsQuantizer(
                                                           node_q_cfg=node.candidates_quantization_cfg,
                                                           float_weights=node.get_weights_by_keys(KERNEL),
                                                           max_candidate_idx=node.find_max_candidates_indices()[0])
                                                   })

        for attr, q in wrapper_layer.weights_quantizers.items():
            self.unit_test.assertEqual(q.active_quantization_config_index, 0)
            # Changing active quantizer candidate index manually to 1 (this is an invalid value in this case)
            q.active_quantization_config_index = 1

        set_layer_to_bitwidth(wrapper_layer, bitwidth_idx=0, weights_quantizer_type=ConfigurableWeightsQuantizer,
                              activation_quantizer_type=ConfigurableActivationQuantizer,
                              weights_quant_layer_type=PytorchQuantizationWrapper,
                              activation_quant_layer_type=PytorchActivationQuantizationHolder)

        for attr, q in wrapper_layer.weights_quantizers.items():
            self.unit_test.assertEqual(q.active_quantization_config_index, 0)


class TestSetLayerToBitwidthActivation(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_inputs_shape(self):
        return [[1, 3, 8, 8]]

    def representative_data_gen(self, n_iters=1):
        input_shapes = self.create_inputs_shape()
        for _ in range(n_iters):
            yield self.generate_inputs(input_shapes)

    def run_test(self, seed=0, **kwargs):
        node, layer = test_setup(self.representative_data_gen)
        holder_layer = \
            PytorchActivationQuantizationHolder(ConfigurableActivationQuantizer(
                node_q_cfg=node.candidates_quantization_cfg,
                max_candidate_idx=node.find_max_candidates_indices()[0]))

        q = holder_layer.activation_holder_quantizer

        self.unit_test.assertEqual(q.active_quantization_config_index, 0)

        # Changing active quantizer candidate index manually to 1 (this is an invalid value in this case)
        q.active_quantization_config_index = 1

        set_layer_to_bitwidth(holder_layer, bitwidth_idx=0, weights_quantizer_type=ConfigurableWeightsQuantizer,
                              activation_quantizer_type=ConfigurableActivationQuantizer,
                              weights_quant_layer_type=PytorchQuantizationWrapper,
                              activation_quant_layer_type=PytorchActivationQuantizationHolder)

        self.unit_test.assertEqual(q.active_quantization_config_index, 0)
