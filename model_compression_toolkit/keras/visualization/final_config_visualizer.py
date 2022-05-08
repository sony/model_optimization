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

import numpy as np
from keras.layers import Conv2D, DepthwiseConv2D, Dense, Conv2DTranspose
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from model_compression_toolkit.common.defaultdict import DefaultDict

from model_compression_toolkit.common import Graph
from model_compression_toolkit.common.similarity_analyzer import compute_cs
from model_compression_toolkit.keras.back2framework.model_builder import model_builder
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.graph.base_node import BaseNode


WEIGHTS_LAYERS_NAMES = {
    Conv2D: 'Conv2D',
    DepthwiseConv2D: 'DW',
    Dense: 'FC',
    Conv2DTranspose: 'Conv2D-T'
}


def get_layer_represent_name(layer_type):
    name = WEIGHTS_LAYERS_NAMES.get(layer_type)
    if name is None:
        raise Exception(f"Layer type {layer_type} is not familiar for config visualization purposes")
    return name


class KerasWeightsConfigVisualizer:

    def __init__(self,
                 final_config: List[Tuple[type, int]]):

        self.final_config = final_config
        self.node_reps_names = [get_layer_represent_name(node_cfg[0]) for node_cfg in self.final_config]
        self.node_final_bitwidth = [node_cfg[1] for node_cfg in self.final_config]
        self.bitwidth_colors_map = {2: 'tomato', 4: 'royalblue', 8: 'limegreen'}
        self.configs_colors = [self.bitwidth_colors_map[b] for b in self.node_final_bitwidth]
        self.bar_width = 2
        self.gap = self.bar_width + 3

    def plot_config_bitwidth(self) -> Figure:
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


class KerasActivationConfigVisualizer:

    def __init__(self,
                 final_config: List[Tuple[type, int]]):
        pass

    def plot_config_bitwidth(self) -> Figure:
        pass