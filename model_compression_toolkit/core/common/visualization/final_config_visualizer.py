# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from model_compression_toolkit.core.common import Graph, BaseNode


def get_kernel_layer_represent_name(node: BaseNode) -> str:
    """
    Returns the mapping between a layer's type and its name to appear in the visualization figure.
    We apply this function only to map types of layers that have kernels (other layers do not appear by name
    in the figures)

    Args:
        node: A graph node representing a model's layer.

    Returns: The name of the layer to appear in the visualization.

    """

    return node.type.__name__


class WeightsFinalBitwidthConfigVisualizer:
    """
    Class to visualize the chosen bit-width configuration for weights configurable layers in mixed-precision mode.
    WeightsFinalBitwidthConfigVisualizer draws a bar plot with the bit-width value of each layer.
    """
    def __init__(self,
                 final_weights_nodes_config: List[Tuple[BaseNode, int]]):
        """
        Initialize a WeightsFinalBitwidthConfigVisualizer object.
        Args:
            final_weights_nodes_config: List of candidates' indices of sorted weights configurable nodes
                (expects a list of tuples - (node, node's final weights bitwidth).
        """

        self.final_weights_nodes_config = final_weights_nodes_config
        self.node_reps_names = [get_kernel_layer_represent_name(node_cfg[0]) for node_cfg in self.final_weights_nodes_config]
        self.node_final_bitwidth = [node_cfg[1] for node_cfg in self.final_weights_nodes_config]
        self.bitwidth_colors_map = {2: 'tomato', 4: 'royalblue', 8: 'limegreen'}
        self.configs_colors = [self.bitwidth_colors_map[b] for b in self.node_final_bitwidth]
        self.bar_width = 2
        self.gap = self.bar_width + 3

    def plot_config_bitwidth(self) -> Figure:
        """
        Plots a bar figure with the layers' bit-width values according to the chosen config.

        Returns:
            Figure of the layers' bit-width values.
        """

        layers_loc = [i for i in range(self.bar_width, self.bar_width + self.gap * len(self.node_reps_names), self.gap)]
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.bar(layers_loc, self.node_final_bitwidth, color=self.configs_colors, width=self.bar_width, align='center')
        plt.xticks(layers_loc, self.node_reps_names, rotation='vertical')
        plt.rc('xtick', labelsize=4)
        plt.rc('ytick', labelsize=4)
        plt.xlabel('Layers', fontsize=12)
        plt.ylabel('Number of bits', fontsize=12)
        plt.tight_layout()
        return fig


class ActivationFinalBitwidthConfigVisualizer:
    """
    Class to visualize the activation configuration attributes.
    ActivationFinalBitwidthConfigVisualizer can draw a figure of the chosen bit-width configuration for activation
    configurable layers in mixed-precision mode.
    It also allows to draw a figure with the activation tensors memory size.
    """

    def __init__(self,
                 final_activation_nodes_config: List[Tuple[BaseNode, int]]):
        """
        Initialize a ActivationFinalBitwidthConfigVisualizer object.
        Args:
            final_activation_nodes_config: List of candidates' indices of sorted activation configurable nodes.
                (expects a list of tuples - (node, node's final activation bitwidth).
        """

        self.final_activation_nodes_config = final_activation_nodes_config
        self.node_final_bitwidth = [node_cfg[1] for node_cfg in self.final_activation_nodes_config]
        self.bar_width = 1
        self.vis_comp_rates = {4.0: 'tomato', 8.0: 'orange', 12.0: 'limegreen'}

    def plot_config_bitwidth(self) -> Figure:
        """
        Plots a bar figure with the layers' bit-width values according to the chosen config.

        Returns:
            Figure of the layers' bit-width values.
        """

        layers_loc = [i for i in range(1, len(self.node_final_bitwidth) + 1)]
        fig, ax = plt.subplots()
        plt.bar(layers_loc, self.node_final_bitwidth, width=self.bar_width, align='center')
        plt.grid()
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Number of bits', fontsize=12)
        plt.tight_layout()
        return fig

    def plot_tensor_sizes(self, graph: Graph) -> Figure:
        """
        Plots a bar figure with the layers' activation tensors sizes.
        Also, adds horizontal line indicators for the max tensor size (in MB) for the defined set of compression rates.

        Args:
            graph: A Graph object to be used for calculating the layers' activation tensor sizes.

        Returns:
            Figure of the layers' activation tensors sizes.
        """

        tensors_sizes = [4.0 * n.get_total_output_params() / 1000000.0
                         for n in graph.get_sorted_activation_configurable_nodes()]  # in MB
        max_tensor_size = max(tensors_sizes)
        max_lines = [(rate, max_tensor_size / rate, color) for rate, color in self.vis_comp_rates.items()]

        layers_loc = [i for i in range(1, len(self.final_activation_nodes_config) + 1)]
        fig, ax = plt.subplots()
        plt.bar(layers_loc, tensors_sizes, width=self.bar_width, align='center')
        plt.grid()
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Tensor Size [MB]', fontsize=12)

        # add max tensor lines
        for rate, t_size, color in max_lines:
            plt.plot([layers_loc[0], layers_loc[-1]], [t_size, t_size], "k--", color=color, label=f"ACR = {rate}")

        plt.legend()
        plt.tight_layout()
        return fig
