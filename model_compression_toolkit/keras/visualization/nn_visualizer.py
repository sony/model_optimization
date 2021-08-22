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


from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from model_compression_toolkit.common import Graph
from model_compression_toolkit.keras.back2framework.model_builder import model_builder, ModelBuilderMode
from model_compression_toolkit.keras.knowledge_distillation.graph_info import get_compare_points


def tensor_norm(x: np.ndarray) -> np.float:
    """
    Compute the L2-norm of a tensor x.
    Args:
        x: Tensor to compute its norm

    Returns:
        L2 norm of x.
    """
    return np.sqrt(np.power(x.flatten(), 2.0).sum())


def cosine_similarity(a: np.ndarray,
                      b: np.ndarray,
                      eps: float = 1e-8) -> np.float:
    """
    Compute the cosine similarity between two tensor.
    Args:
        a: First tensor to compare.
        b: Second tensor to compare.
        eps: Small value to avoid zero division.

    Returns:
        The cosine similarity between two tensors.
    """

    if np.all(b == 0) and np.all(a == 0):
        return 1.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = tensor_norm(a)
    b_norm = tensor_norm(b)

    return np.sum(a_flat * b_flat) / ((a_norm * b_norm) + eps)


class KerasNNVisualizer(object):
    """
    Class to build two models from two graph: a float and a quantized version.
    KerasNNVisualizer can compare the two models outputs after each layer.
    The results can be displayed using plot_cs_graph by passing it an input image to use
    for inference of the models.
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quantized: Graph):
        """
        Initialize a KerasNNVisualizer object.
        Args:
            graph_float: Float version of the graph.
            graph_quantized: Quantized version of the graph.

        """

        self.graph_float = graph_float
        self.graph_quantized = graph_quantized

        # Get compare points of two graphs.
        self.compare_points, self.compare_points_name = get_compare_points(self.graph_quantized)
        self.compare_points_float, self.compare_points_name_float = get_compare_points(self.graph_float)

        self.keras_graph_quantized, _ = model_builder(self.graph_quantized,
                                                      mode=ModelBuilderMode.QUANTIZED,
                                                      append2output=self.compare_points)
        self.keras_graph_float, _ = model_builder(self.graph_float,
                                                  mode=ModelBuilderMode.FLOAT,
                                                  append2output=self.compare_points_float)

    def plot_cs_graph(self, input_image: np.ndarray,
                      y_limits: Tuple[float, float] = (0.5, 1.0)) -> Figure:
        """
        Compare and plot the outputs of the quantized and the float versions
        of a neural network that KerasNNVisualizer has.

        Args:
            input_image: Image to use as input to the networks.
            y_limits: Limits for y axis of the plot.

        Returns:
            Figure of the cosine similarity per layer.
        """

        # To compare cosine similarity, we use a single image as input (per input),
        # to make the difference more noticeable when exists
        new_inputs = []
        for single_input in input_image:
            img = single_input[0]
            new_inputs.append(np.expand_dims(img, axis=0))

        # Get outputs
        tensors_float = self.keras_graph_float(new_inputs)
        tensors_fxp = self.keras_graph_quantized(new_inputs)

        # Compute cosine similarities between couples of outputs.
        cs_array = np.asarray(
            [cosine_similarity(t_float.numpy(), t_fxp.numpy()) for t_float, t_fxp in zip(tensors_float, tensors_fxp)])

        # Display the result: cosine similarity at every layer's output.
        fig = plt.figure()
        plt.plot(cs_array)
        plt.ylim(y_limits)
        plt.grid()
        plt.xlabel('Layer')
        plt.ylabel('Cosine Similarity')
        return fig
