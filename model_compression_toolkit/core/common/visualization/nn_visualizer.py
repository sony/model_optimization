# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import Tuple, List, Callable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_node import BaseNode
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.similarity_analyzer import compute_cs


def _get_compare_points(input_graph: Graph) -> Tuple[List[BaseNode], List[str]]:
    """
    Create a list of nodes in a graph where we collect their output statistics, and a corresponding list
    of their names for tensors comparison purposes.
    Args:
        input_graph: Graph to get its points to compare.

    Returns:
        A list of nodes in a graph, and a list of their names.
    """
    compare_points = []
    compare_points_name = []
    for n in input_graph.get_topo_sorted_nodes():
        tensors = input_graph.get_out_stats_collector(n)
        if (not isinstance(tensors, list)) and tensors.require_collection():
            compare_points.append(n)
            compare_points_name.append(n.name)
    return compare_points, compare_points_name


class NNVisualizer:
    """
    Class to build two models from two graph: a float and a quantized version.
    NNVisualizer can compare the two models outputs after each layer.
    The results can be displayed using plot_cs_graph by passing it an input image to use
    for inference of the models.
    """

    def __init__(self,
                 graph_float: Graph,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo):
        """
        Initialize a NNVisualizer object.
        Args:
            graph_float: Float version of the graph.

        """

        self.graph_float = graph_float
        self.graph_quantized = quantize_graph_weights(graph_float, fw_info=fw_info, fw_impl=fw_impl)
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # Get compare points of two graphs.
        self.compare_points, self.compare_points_name = _get_compare_points(self.graph_quantized)
        self.compare_points_float, self.compare_points_name_float = _get_compare_points(self.graph_float)

        self.quantized_model, _ = self.fw_impl.model_builder(self.graph_quantized,
                                                             mode=ModelBuilderMode.QUANTIZED,
                                                             append2output=self.compare_points,
                                                             fw_info=self.fw_info)

        self.float_model, _ = self.fw_impl.model_builder(self.graph_float,
                                                         mode=ModelBuilderMode.FLOAT,
                                                         append2output=self.compare_points_float,
                                                         fw_info=self.fw_info)

    def plot_distance_graph(self,
                            input_image: np.ndarray,
                            distance_fn: Callable = compute_cs,
                            convert_to_range: Callable = lambda a: a) -> Figure:
        """
        Compare and plot the outputs of the quantized and the float versions
        of a neural network that KerasNNVisualizer has.

        Args:
            input_image: Image to use as input to the networks.
            distance_fn: Distance function to calculate the distance between two tensors.
            convert_to_range: Optional function to move the distance values into a specific range, e.g., when using
                cosine similarity for distance, use 'lambda a: 1 - 2 * a' to convert the distance values to the range
                of [-1, 1].

        Returns:
            Figure of the distance per layer.
        """

        # To compare cosine similarity, we use a single image as input (per input),
        # to make the difference more noticeable when exists
        new_inputs = []
        for single_input in input_image:
            img = single_input[0]
            new_inputs.append(np.expand_dims(img, axis=0))

        # Get outputs
        tensors_float = self.fw_impl.run_model_inference(self.float_model, new_inputs)
        tensors_fxp = self.fw_impl.run_model_inference(self.quantized_model, new_inputs)

        # Compute distance between couples of outputs.
        distance_array = np.asarray(
            [distance_fn(self.fw_impl.to_numpy(t_float), self.fw_impl.to_numpy(t_fxp)) for t_float, t_fxp in zip(tensors_float, tensors_fxp)])

        distance_array = convert_to_range(distance_array)

        # Display the result: distance at every layer's output.
        fig = plt.figure()
        plt.plot(distance_array)
        eps = 0.5
        y_limits = (min(distance_array) - eps, max(distance_array) + eps)
        plt.ylim(y_limits)
        plt.grid()
        plt.xlabel('Layer')
        plt.ylabel('Distance')
        return fig
