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
from typing import Callable, Any

import copy

import abc

import pytest


from model_compression_toolkit.core import QuantizationConfig, FrameworkInfo
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfo
from model_compression_toolkit.core.common.fusion.fusing_metadata_wrapper import FusingMetadataWrapper
from model_compression_toolkit.core.common.graph.edge import EDGE_SOURCE_INDEX, EDGE_SINK_INDEX
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner


class BaseFusingInfoGeneratorTest(abc.ABC):

    fw_impl: FrameworkImplementation
    fw_info: FrameworkInfo
    attach_to_fw_func: Callable
    expected_fi: FusingInfo

    def _data_gen(self):
        raise NotImplementedError()

    def _get_model(self):
        raise NotImplementedError()

    def _get_tpc(self, default_quant_cfg_options):
        raise NotImplementedError()

    def _get_qc(self):
        raise NotImplementedError()

    @pytest.fixture
    def graph_with_fusion_metadata(self, default_quant_cfg_options):
        """
        Creates a graph with fusing metadata based on a generated model and a predefined configuration.
        Ensures all required components (framework implementation, framework info, etc.) are present.
        """
        assert self._data_gen is not None
        assert self.fw_impl is not None
        assert self.fw_info is not None
        assert self.attach_to_fw_func is not None
        assert self.expected_fi is not None

        self.fqc = self.attach_to_fw_func(self._get_tpc(default_quant_cfg_options),
                                          self._get_qc().custom_tpc_opset_to_layer)

        graph_with_fusion_metadata = graph_preparation_runner(self._get_model(),
                                                              self._data_gen,
                                                              self._get_qc(),
                                                              fw_info=self.fw_info,
                                                              fw_impl=self.fw_impl,
                                                              fqc=self.fqc,
                                                              mixed_precision_enable=False,
                                                              running_gptq=False)
        return graph_with_fusion_metadata

    def test_expected_fusing_info(self, graph_with_fusion_metadata: FusingMetadataWrapper):
        actual_fi = graph_with_fusion_metadata.get_fusing_info()
        assert self.expected_fi.node_to_fused_node_map == actual_fi.node_to_fused_node_map






