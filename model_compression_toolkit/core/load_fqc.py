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
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.fusion.fusing_info import FusingInfoGenerator
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.target_platform_capabilities import FrameworkQuantizationCapabilities


def load_fqc_configuration(graph: Graph, fqc: FrameworkQuantizationCapabilities):
    """
    Set-up graph for quantization per TPC.
    Each node will contain quantization candidates for mixed precision and the base config for single precision.
    The graph will contain fusing info.

    Args:
        graph: graph.
        fqc: framework quantization capabilities object.

    Returns:
        Updated graph.
    """
    graph = set_quantization_configuration_to_graph(graph=graph, fqc=fqc)

    # TODO fix the horrible dict with const keys inside get_fusing_patterns. use named tuple or class
    fusing_info = FusingInfoGenerator(fqc.get_fusing_patterns()).generate_fusing_info(graph)
    graph.fusing_info = fusing_info
    graph.disable_fused_nodes_activation_quantization()

    return graph
