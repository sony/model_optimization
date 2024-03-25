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
from typing import Tuple, List

from model_compression_toolkit.core import FrameworkInfo
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.graph.base_node import BaseNode


def get_compare_points(input_graph: Graph) -> Tuple[List[BaseNode], List[str], List, List]:
    """
    Create a list of nodes with weights in a graph and a corresponding list
    of their names for tensors comparison purposes. Also outputs 2 list of activations
    prior information collected from batch normalization nodes (if exists)
    Args:
        input_graph: Graph to get its points to compare.

    Returns:
        A list of nodes in a graph
        A list of their names.
        A list of nodes mean collected from BatchNorms in the graph
        A list of nodes std collected from BatchNorms in the graph
    """
    compare_points = []
    compare_points_mean = []
    compare_points_std = []
    compare_points_name = []
    for n in input_graph.get_topo_sorted_nodes():
        # only nodes with kernel attribute are currently trained with GPTQ and are used as compare points
        kernel_attr = input_graph.fw_info.get_kernel_op_attributes(n.type)[0]
        if kernel_attr is not None and n.is_weights_quantization_enabled(kernel_attr) and not n.reuse:
            compare_points.append(n)
            compare_points_name.append(n.name)
            compare_points_std.append(n.prior_info.std_output)
            compare_points_mean.append(n.prior_info.mean_output)
    return compare_points, compare_points_name, compare_points_mean, compare_points_std


def get_kernel_attribute_name_for_gptq(layer_type: type, fw_info: FrameworkInfo) -> str:
    """
    Returns a layer's kernel attribute name for GPTQ training purposes.

    Args:
        layer_type: A type of model's layer.
        fw_info: A FrameworkInfo object.

    Returns: The name of the kernel attribute.

    """
    kernel_attribute = fw_info.get_kernel_op_attributes(layer_type)
    if len(kernel_attribute) != 1:
        Logger.critical(  # pragma: no cover
            f"In GPTQ training, only the kernel weights attribute should be trained. "
            f"However, the number of kernel attributes is {len(kernel_attribute)}.")
    return kernel_attribute[0]
