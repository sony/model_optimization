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
import numpy as np
import torch
from model_compression_toolkit.core.common.quantization.quantization_params_generation.qparams_computation import \
    calculate_quantization_params
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import ActivationQuantizationMode
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import OpQuantizationConfig
from model_compression_toolkit.core import QuantizationConfig, QuantizationErrorMethod
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import Signedness, \
    AttributeQuantizationConfig
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.core.common.collectors.statistics_collector import StatsCollector
from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, WEIGHTS_N_BITS
from mct_quantizers import QuantizationMethod

class TestCalculateQuantizationParams:
    def get_op_qco(self):
        # define a default quantization config for all non-specified weights attributes.
        default_weight_attr_config = AttributeQuantizationConfig()

        # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
        kernel_base_config = AttributeQuantizationConfig(
            weights_n_bits=8,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True)

        base_cfg = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={KERNEL_ATTR: kernel_base_config},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            signedness=Signedness.AUTO)

        default_config = schema.OpQuantizationConfig(
            default_weight_attr_config=default_weight_attr_config,
            attr_weights_configs_mapping={},
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            signedness=Signedness.AUTO
        )

        mx_cfg_list = [base_cfg]
        for n in [8, 4, 2]:
            mx_cfg_list.append(base_cfg.clone_and_edit(attr_to_edit={KERNEL_ATTR: {WEIGHTS_N_BITS: n}}))

        return base_cfg, mx_cfg_list, default_config

    def generate_tpc_local(self, default_config, base_config, mixed_precision_cfg_list):
        default_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple([default_config]))
        mixed_precision_configuration_options = schema.QuantizationConfigOptions(
            quantization_configurations=tuple(mixed_precision_cfg_list),
            base_config=base_config)

        operator_set = []

        conv = schema.OperatorsSet(name=schema.OperatorSetNames.CONV, qc_options=mixed_precision_configuration_options)
        relu = schema.OperatorsSet(name=schema.OperatorSetNames.RELU)
        operator_set.extend([conv, relu])

        generated_tpc = schema.TargetPlatformCapabilities(
            default_qco=default_configuration_options,
            operator_set=tuple(operator_set))

        return generated_tpc

    def get_tpc(self):
        base_cfg, mx_cfg_list, default_config = self.get_op_qco()
        tpc = self.generate_tpc_local(default_config, base_cfg, mx_cfg_list)
        return tpc

    def get_float_model(self):
        class BaseModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.conv3(x)
                return x

        return BaseModel()

    def _create_weights_attr_quantization_config(self, weights_n_bits: int) -> AttributeQuantizationConfig:
        weights_attr_config = AttributeQuantizationConfig(weights_n_bits=weights_n_bits)
        return weights_attr_config

    def _create_node_weights_op_cfg(self,
                                    def_weight_attr_config: AttributeQuantizationConfig) -> OpQuantizationConfig:
        # define a quantization config to quantize the kernel (for layers where there is a kernel attribute).
        kernel_base_config = AttributeQuantizationConfig(
            weights_quantization_method=QuantizationMethod.SYMMETRIC,
            enable_weights_quantization=True,
            weights_n_bits=8)

        # define a quantization config to quantize the bias (for layers where there is a bias attribute).
        bias_config = AttributeQuantizationConfig()

        attr_weights_configs_mapping = {'weight': kernel_base_config, 'bias': bias_config}
        op_cfg = OpQuantizationConfig(
            default_weight_attr_config=def_weight_attr_config,
            attr_weights_configs_mapping=attr_weights_configs_mapping,
            activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
            activation_n_bits=8,
            supported_input_activation_n_bits=8,
            enable_activation_quantization=True,
            quantization_preserving=False,
            signedness=Signedness.AUTO
        )

        return op_cfg

    def get_test_graph(self, qem: QuantizationErrorMethod):
        float_model = self.get_float_model()
        fw_info = DEFAULT_PYTORCH_INFO

        fw_impl = PytorchImplementation()
        graph = fw_impl.model_reader(float_model,
                                     self.representative_data_gen)
        graph.set_fw_info(fw_info)

        quantization_config = QuantizationConfig(weights_error_method=qem)

        tpc = self.get_tpc()
        attach2pytorch = AttachTpcToPytorch()
        fqc = attach2pytorch.attach(
            tpc, quantization_config.custom_tpc_opset_to_layer)
        graph.set_fqc(fqc)

        def_weight_attr_config = self._create_weights_attr_quantization_config(weights_n_bits=8)
        op_cfg = self._create_node_weights_op_cfg(def_weight_attr_config=def_weight_attr_config)

        graph.node_to_out_stats_collector = dict()
        for id, n in enumerate(graph.nodes):
            n.prior_info = fw_impl.get_node_prior_info(node=n, fw_info=fw_info, graph=graph)
            n.candidates_quantization_cfg = []
            candidate_qc_a = CandidateNodeQuantizationConfig(
                activation_quantization_cfg=NodeActivationQuantizationConfig(qc=quantization_config, op_cfg=op_cfg,
                                                                             activation_quantization_fn=None,
                                                                             activation_quantization_params_fn=None),
                weights_quantization_cfg=NodeWeightsQuantizationConfig(qc=quantization_config, op_cfg=op_cfg,
                                                                       weights_channels_axis=(0, 1),
                                                                       node_attrs_list=['weight', 'bias'])
            )
            if n.name in ['conv3']:
                candidate_qc_a.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.FLN_QUANT
                candidate_qc_a.activation_quantization_cfg.activation_n_bits = 16 # set 16bit for FLN node for test.
            else:
                candidate_qc_a.activation_quantization_cfg.quant_mode = ActivationQuantizationMode.QUANT
            n.candidates_quantization_cfg.append(candidate_qc_a)

            graph.node_to_out_stats_collector[n] = StatsCollector(init_min_value=0.0, init_max_value=1.0, out_channel_axis=fw_info.out_channel_axis_mapping.get(n.type))
            graph.node_to_out_stats_collector[n].hc._n_bins = 3
            if n.name in ['conv1']:
                graph.node_to_out_stats_collector[n].hc._bins = np.array([0.4, 0.8, 1.2])
            elif n.name in ['conv2']:
                graph.node_to_out_stats_collector[n].hc._bins = np.array([0.7, 1.4, 2.1])
            elif n.name in ['conv3']:
                graph.node_to_out_stats_collector[n].hc._bins = np.array([-32, -24, -1])
            elif n.name in ['relu']:
                graph.node_to_out_stats_collector[n].hc._bins = np.array([2.0, 4.0, 6.0])
            else:
                graph.node_to_out_stats_collector[n].hc._bins = np.array([0.1, 0.2, 0.3])
            graph.node_to_out_stats_collector[n].hc._counts = np.array([1, 1])

        return graph, fw_impl

    def representative_data_gen(self, shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=10):
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

    def test_calculate_quantization_params(self):
        graph, fw_impl = self.get_test_graph(QuantizationErrorMethod.MSE)

        calculate_quantization_params(graph, fw_impl, self.representative_data_gen)

        for node in graph.nodes:
            for candidate_qc in node.candidates_quantization_cfg:
                assert type(candidate_qc.activation_quantization_cfg.activation_quantization_params) == dict
                assert 'threshold' in candidate_qc.activation_quantization_cfg.activation_quantization_params.keys()
                assert 'is_signed' in candidate_qc.activation_quantization_cfg.activation_quantization_params.keys()

                threshold = candidate_qc.activation_quantization_cfg.activation_quantization_params['threshold']
                is_signed = candidate_qc.activation_quantization_cfg.activation_quantization_params['is_signed']
                if node.name in 'conv1':
                    assert threshold == 1.0
                    assert is_signed == False
                elif node.name in 'conv2':
                    assert threshold == 2.0
                    assert is_signed == False
                elif node.name in 'conv3':
                    assert threshold == 64.0
                    assert is_signed == True
                elif node.name in 'relu':
                    assert threshold == 16.0
                    assert is_signed == False
