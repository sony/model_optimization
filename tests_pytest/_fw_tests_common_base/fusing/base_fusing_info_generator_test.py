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
import random
from typing import Callable, List

import abc
from unittest.mock import Mock

import pytest
from mct_quantizers import QuantizationMethod

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
from model_compression_toolkit.core.common.fusion.graph_fuser import GraphFuser
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, ActivationQuantizationMode
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from tests_pytest._test_util.tpc_util import minimal_cfg_options


def random_activation_configs():
    num_candidates = random.choice([1, 2, 3])
    bits_list = random.sample(range(2, 9), k=num_candidates)
    qcs = [
        CandidateNodeQuantizationConfig(
            weights_quantization_cfg=None,
            activation_quantization_cfg=Mock(spec=NodeActivationQuantizationConfig,
                                             activation_n_bits=nb, quant_mode=ActivationQuantizationMode.QUANT)
        )
        for nb in bits_list
    ]
    return bits_list, qcs


def get_activation_mp_options(last_node_activation_nbits):
    options = tuple([schema.OpQuantizationConfig(
        default_weight_attr_config={},
        attr_weights_configs_mapping={},
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        activation_n_bits=a_nbits,
        supported_input_activation_n_bits=[8],
        enable_activation_quantization=True,
        quantization_preserving=False,
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=32,
        signedness=schema.Signedness.AUTO) for a_nbits in last_node_activation_nbits])

    cfg_options = schema.QuantizationConfigOptions(quantization_configurations=options, base_config=options[0])

    return cfg_options


class BaseFusingInfoGeneratorTest(abc.ABC):

    fw_impl: FrameworkImplementation
    fw_info: FrameworkInfo
    attach_to_fw_func: Callable
    expected_fi: FusingInfo
    last_node_activation_nbits: List[int]

    def _data_gen(self):
        raise NotImplementedError()

    def _get_model(self):
        raise NotImplementedError()

    def _get_tpc(self, default_quant_cfg_options):
        raise NotImplementedError()

    def _get_qc(self):
        raise NotImplementedError()

    @pytest.fixture
    def graph_with_fusion_metadata(self):
        """
        Creates a graph with fusing metadata based on a generated model and a predefined configuration.
        Ensures all required components (framework implementation, framework info, etc.) are present.
        """
        self.fqc = self.attach_to_fw_func(self._get_tpc(minimal_cfg_options()),
                                          self._get_qc().custom_tpc_opset_to_layer)

        graph_with_fusion_metadata = graph_preparation_runner(self._get_model(),
                                                              self._data_gen,
                                                              self._get_qc(),
                                                              fw_impl=self.fw_impl,
                                                              fqc=self.fqc,
                                                              mixed_precision_enable=True,
                                                              running_gptq=False)
        return graph_with_fusion_metadata

    @pytest.fixture
    def fused_graph(self, graph_with_fusion_metadata):
        return GraphFuser().apply_node_fusion(graph_with_fusion_metadata)

    def test_expected_fusing_info(self, graph_with_fusion_metadata):
        actual_fi = graph_with_fusion_metadata.fusing_info
        assert self.expected_fi.node_name_to_fused_op_id == actual_fi.node_name_to_fused_op_id

    def test_expected_fused_graph(self, fused_graph):
        expected_fused_nodes = self.expected_fi.fusing_data
        graph_node_names = [node.name for node in fused_graph.nodes]

        for fused_node_name, original_nodes in expected_fused_nodes.items():
            # 1. Fused node must exist
            assert fused_node_name in graph_node_names, f"Fused node '{fused_node_name}' not found in graph."
            fused_node = fused_graph.find_node_by_name(fused_node_name)
            assert len(fused_node) == 1, f"Expected to find a single node, but found {len(fused_node)}"
            fused_node = fused_node[0]

            # 2. Original nodes should not exist anymore
            for node in original_nodes:
                assert node.name not in graph_node_names, (
                    f"Original node '{node.name}' should be fused into '{fused_node_name}', "
                    f"but it's still in the graph."
                )

            # 3. Final quantization configs
            if original_nodes[0].final_weights_quantization_cfg is not None:
                assert fused_node.final_weights_quantization_cfg == original_nodes[0].final_weights_quantization_cfg, (f"Incorrect final_weights_quantization_cfg for '{fused_node_name}'")

            if original_nodes[-1].final_activation_quantization_cfg is not None:
                assert fused_node.final_activation_quantization_cfg == original_nodes[-1].final_activation_quantization_cfg, (f"Incorrect final_activation_quantization_cfg for '{fused_node_name}'")

            # 4. Candidate quantization configs
            expected_candidates = original_nodes[-1].candidates_quantization_cfg
            actual_candidates = fused_node.candidates_quantization_cfg

            assert len(actual_candidates) == len(expected_candidates), (
                f"Mismatch in number of candidate quantization configs for '{fused_node_name}'")

            # Extract and sort the n_bits values for comparison
            actual_nbits = sorted([c.activation_quantization_cfg.activation_n_bits for c in actual_candidates])
            expected_nbits = sorted([c.activation_quantization_cfg.activation_n_bits for c in expected_candidates])

            assert actual_nbits == expected_nbits, (
                f"Mismatch in activation_n_bits list for '{fused_node_name}': "
                f"expected {expected_nbits}, got {actual_nbits}")

            # Optionally also assert all weights quant configs are None
            for i, actual in enumerate(actual_candidates):
                assert actual.weights_quantization_cfg is None, (
                    f"Weights quant config should be None in fused candidate #{i} for '{fused_node_name}'")







