# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import pytest

#import model_compression_toolkit as mct
from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core import CoreConfig

from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph

import torch
from torch import nn
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_op_quantization_configs, generate_tpc

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR, WEIGHTS_N_BITS
from model_compression_toolkit.target_platform_capabilities.constants import PYTORCH_KERNEL, BIAS

class TestManualWeightsBitwidthSelection:
    def get_tpc(self):
        base_cfg, _, default_config = get_op_quantization_configs()

        mx_cfg_list = [base_cfg]
        for n in [2, 4, 16]:
            mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))
            mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={BIAS_ATTR: {WEIGHTS_N_BITS: n}}))
        mx_cfg_list.append(
            base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: 4}, BIAS_ATTR: {WEIGHTS_N_BITS: 16}})
        )
        tpc = generate_tpc(default_config=default_config, base_config=base_cfg, mixed_precision_cfg_list=mx_cfg_list,
                           name='imx500_tpc_kai')

        return tpc

    def representative_data_gen(self, shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

    def get_float_model(self):
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                return x
        return BaseModel()

    def get_test_graph(self, core_config):
        float_model = self.get_float_model()
        fw_info = DEFAULT_PYTORCH_INFO

        fw_impl = PytorchImplementation()
        graph = fw_impl.model_reader(float_model,
                                     self.representative_data_gen)
        graph.set_fw_info(fw_info)

        tpc = self.get_tpc()
        attach2pytorch = AttachTpcToPytorch()
        fqc = attach2pytorch.attach(
            tpc, core_config.quantization_config.custom_tpc_opset_to_layer)
        graph.set_fqc(fqc)

        return graph

    # test case for set_manual_activation_bit_width
    """
    Test Items Policy:
        - How to specify the target layer: Options(type/name)
        - Target attribute information: Options(kernel/bias) 
        - Bit width variations: Options(2, 4, 16)
    """
    test_input_1 = (NodeNameFilter("conv1"), 2, PYTORCH_KERNEL)
    test_input_2 = (NodeTypeFilter(nn.Conv2d), 16, PYTORCH_KERNEL)
    test_input_3 = ([NodeNameFilter("conv1"), NodeNameFilter("conv1")], [4, 16], [PYTORCH_KERNEL, BIAS])

    test_expected_1 = ({"conv1": {PYTORCH_KERNEL: 2, BIAS: 32}, "conv2": {PYTORCH_KERNEL: 8, BIAS: 32}})
    test_expected_2 = ({"conv1": {PYTORCH_KERNEL: 16, BIAS: 32}, "conv2": {PYTORCH_KERNEL: 16, BIAS: 32}})
    test_expected_3 = ({"conv1": {PYTORCH_KERNEL: 4, BIAS: 16}, "conv2": {PYTORCH_KERNEL: 8, BIAS: 32}})

    @pytest.mark.parametrize(
        ("inputs", "expected"), [
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_manual_weights_bitwidth_selection(self, inputs, expected):
        core_config = CoreConfig()
        graph = self.get_test_graph(core_config)

        core_config.bit_width_config.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])

        updated_graph = set_quantization_configuration_to_graph(
            graph, core_config.quantization_config, core_config.bit_width_config,
            False, False
        )

        for node in updated_graph.nodes:
            exp_vals = expected.get(node.name)
            if exp_vals is None: continue
            assert len(node.candidates_quantization_cfg) == 1

            for vkey in node.candidates_quantization_cfg[0].weights_quantization_cfg.attributes_config_mapping:
                cfg = node.candidates_quantization_cfg[0].weights_quantization_cfg.attributes_config_mapping[vkey]
                assert cfg.weights_n_bits == exp_vals[vkey]
