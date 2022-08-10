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
        if len(n.weights) > 0 and n.is_weights_quantization_enabled():
            compare_points.append(n)
            compare_points_name.append(n.name)
            compare_points_std.append(n.prior_info.std_output)
            compare_points_mean.append(n.prior_info.mean_output)
    return compare_points, compare_points_name, compare_points_mean, compare_points_std
