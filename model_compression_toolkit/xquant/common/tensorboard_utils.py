#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo


from model_compression_toolkit.core.common.visualization.tensorboard_writer import TensorboardWriter
from model_compression_toolkit.xquant.common.constants import TENSORBOARD_DEFAULT_TAG
from model_compression_toolkit.logger import Logger


from typing import Any, Dict, Callable


class TensorboardUtils:
    """
    Utility class for handling Tensorboard operations like adding graph to display
    and histograms on the float model.
    """

    def __init__(self,
                 report_dir: str,
                 fw_info: FrameworkInfo,
                 fw_impl: FrameworkImplementation):
        """
        Initialize the TensorboardUtils.

        Args:
            report_dir (str): Directory where Tensorboard logs will be stored.
            fw_info (FrameworkInfo): Framework-specific information.
            fw_impl (FrameworkImplementation): Framework-specific implementation.
        """
        self.fw_impl = fw_impl
        self.fw_info = fw_info
        self.tb_writer = TensorboardWriter(report_dir, fw_info)
        Logger.info(f"Please run: tensorboard --logdir {self.tb_writer.dir_path}")

    def get_graph_for_tensorboard_display(self,
                                          quantized_model: Any,
                                          similarity_metrics: Dict[str, Any],
                                          repr_dataset: Callable) -> Graph:
        """
        Get the graph for Tensorboard display. The framework-specific implementations
        (like KerasTensorboardUtils and PytorchTensorboardUtils) should implement this
        as it differs between them when combining the similarity metrics into the graph.

        Args:
            quantized_model (Any): The quantized model.
            similarity_metrics (Dict[str, Any]): Metrics for model similarity.
            repr_dataset (Callable): Representative dataset function.

        Returns:
            Graph: The generated graph for Tensorboard display.
        """
        Logger.critical("This method should be implemented by the framework-specific TensorboardUtils.") # pragma: no cover

    def add_histograms_to_tensorboard(self,
                                      graph: Graph):
        """
        Add histograms to Tensorboard from a graph that holds these statistics.

        Args:
            graph (Graph): Graph with histograms to add to the tensorboard.
        """
        self.tb_writer.add_histograms(graph, TENSORBOARD_DEFAULT_TAG)

    def add_graph_to_tensorboard(self,
                                 quantized_model: Any,
                                 similarity_metrics: Dict[str, Any],
                                 repr_dataset: Callable):
        """
        Add a graph to Tensorboard. The graph represents the quantized graph
        with the similarity metrics that were measured in different nodes.

        Args:
            quantized_model (Any): The quantized model.
            similarity_metrics (Dict[str, Any]): The similarity metrics that were collected.
            repr_dataset (Callable): Representative dataset to use (if needed, like in pytorch case).
        """
        # Generate the quantized graph with similarity metrics.
        tb_graph = self.get_graph_for_tensorboard_display(quantized_model=quantized_model,
                                                          similarity_metrics=similarity_metrics,
                                                          repr_dataset=repr_dataset)

        self.tb_writer.add_graph(tb_graph, TENSORBOARD_DEFAULT_TAG)


