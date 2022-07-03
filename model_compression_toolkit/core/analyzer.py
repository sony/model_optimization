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


from typing import Callable

from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common import FrameworkInfo
from model_compression_toolkit.core.common.constants import NUM_SAMPLES_DISTANCE_TENSORBOARD
from model_compression_toolkit.core.common.graph.base_graph import Graph

from model_compression_toolkit.core.common.similarity_analyzer import compute_cs
from model_compression_toolkit.core.common.visualization.nn_visualizer import NNVisualizer

from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter


def analyzer_model_quantization(representative_data_gen: Callable,
                                tb_w: TensorboardWriter,
                                tg: Graph,
                                fw_impl: FrameworkImplementation,
                                fw_info: FrameworkInfo):
    """
    Plot the cosine similarity of different points on the graph between the float and quantized
    graphs. Add them to the passed TensorboardWriter object and close all tensorboard writer open
    files.

    Args:
        representative_data_gen: Dataset used for calibration.
        tb_w: TensorBoardWriter object to log events.
        tg: Graph of quantized model.
        fw_impl: FrameworkImplementation object with a specific framework methods implementation.
        fw_info: Information needed for quantization about the specific framework.

    """
    if tb_w is not None:
        visual = NNVisualizer(tg,
                              fw_impl=fw_impl,
                              fw_info=fw_info)

        for i in range(NUM_SAMPLES_DISTANCE_TENSORBOARD):
            figure = visual.plot_distance_graph(representative_data_gen(),
                                                distance_fn=compute_cs,
                                                convert_to_range=lambda a: 1 - 2 * a)
            tb_w.add_figure(figure, f'similarity_distance_sample_{i}')
        tb_w.close()
