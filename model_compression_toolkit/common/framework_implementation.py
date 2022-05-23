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
from abc import ABC, abstractmethod
from typing import Callable, Any, List, Tuple, Type

import numpy as np

from model_compression_toolkit import common, GradientPTQConfig, MixedPrecisionQuantizationConfig
from model_compression_toolkit.common import BaseNode
from model_compression_toolkit.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.graph.base_graph import Graph
from model_compression_toolkit.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.core_config import CoreConfig
from model_compression_toolkit.common.user_info import UserInformation


class FrameworkImplementation(ABC):
    """
    An abstract class with abstract methods that should be implemented when supporting a new
    framework in MCT.
    """

    @property
    def constants(self):
        """

        Returns: Module of the framework constants.

        """
        raise Exception(f'{self.__class__.__name__} did not supply a constants module.')

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert framework's tensor to a Numpy array.
        Args:
            tensor: Framework's tensor.

        Returns:
            Numpy array converted from the input tensor.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s to_numpy method.')    \

    @abstractmethod
    def to_tensor(self, tensor: np.ndarray) -> Any:
        """
        Convert a Numpy array to a framework's tensor.
        Args:
            tensor: Numpy array.

        Returns:
            Framework's tensor converted from the input Numpy array.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s to_tensor method.')

    @abstractmethod
    def model_reader(self,
                     model: Any,
                     representative_data_gen: Callable) -> Graph:
        """
        Convert a framework's model into a graph.
        Args:
            model: Framework's model.
            representative_data_gen (Callable): Dataset used for calibration.

        Returns:
            Graph representing the input model.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s model_reader method.')

    @abstractmethod
    def model_builder(self,
                      graph: Graph,
                      mode: ModelBuilderMode,
                      append2output: List[Any],
                      fw_info: FrameworkInfo,
                      gptq_config: GradientPTQConfig) -> Tuple[Any, UserInformation]:
        """
        Build a framework model from a graph.
        The mode determines how the model should be build. append2output is a list of Nodes
        to set as the model outputs.

        Args:
            graph: Graph to build the model from it.
            mode: Mode for how to build the model.
            append2output: List of Nodes to set as the model's outputs.
            fw_info: FrameworkInfo object with information about the specific framework's model
            gptq_config: GPTQ configuration class

        Returns:
            A tuple of the model that was built and an UserInformation object.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s model_builder method.')

    @abstractmethod
    def run_model_inference(self,
                            model: Any,
                            input_list: List[Any]) -> Tuple[Any]:
        """
        Run the model logic on the given the inputs.

        Args:
            model: Framework's model.
            input_list: List of inputs for the model.

        Returns:
            The frameworks model's output.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s run_model_inference method.')

    @abstractmethod
    def shift_negative_correction(self,
                                  graph: Graph,
                                  core_config: CoreConfig,
                                  fw_info: FrameworkInfo) -> Graph:
        """
        Apply shift negative correction (SNC) on a graph.

        Args:
            graph: Graph to apply SNC on.
            qc: Quantization configuration.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Graph after SNC.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s apply_shift_negative_correction method.')

    @abstractmethod
    def attach_sc_to_node(self, node: BaseNode, output_channel_index: int) -> BaseStatsCollector:
        """
        Return a statistics collector that should be attached to a node's output
        during statistics collection.

        Args:
            node: Node to return its collector.
            output_channel_index: Index of output channels (for statistics per-channel).

        Returns:
            Statistics collector for the node.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s attach_sc_to_node method.')

    @abstractmethod
    def get_substitutions_marking(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used for marking
        points we fuse.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_substitutions_marking method.')

    @abstractmethod
    def get_substitutions_channel_equalization(self,
                                               quant_config: QuantizationConfig,
                                               fw_info: FrameworkInfo) -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used for channel equalization.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_substitutions_channel_equalization method.')

    @abstractmethod
    def get_substitutions_prepare_graph(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used to prepare the graph.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_substitutions_prepare_graph method.')

    @abstractmethod
    def get_substitutions_pre_statistics_collection(self, quant_config: QuantizationConfig) -> \
            List[common.BaseSubstitution]:
        """

        Args:
            quant_config: Quantization configuration.

        Returns: A list of the framework substitutions used before we collect statistics.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_substitutions_pre_statistics_collection method.')

    @abstractmethod
    def get_linear_collapsing_substitution(self) -> common.BaseSubstitution:
        """
        Returns: linear collapsing substitution
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_linear_collapsing_substitution method.')

    @abstractmethod
    def get_residual_collapsing_substitution(self) -> List[common.BaseSubstitution]:
        """
        Returns: A list of the framework substitutions used for residual collapsing
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_residual_collapsing_substitution method.')

    @abstractmethod
    def get_substitutions_pre_build(self) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used before we build a quantized model.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_substitutions_pre_build method.')

    @abstractmethod
    def get_substitutions_post_statistics_collection(self, quant_config: QuantizationConfig) -> List[
        common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used after we collect statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used after we collect statistics.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_substitutions_post_statistics_collection method.')

    @abstractmethod
    def get_gptq_trainer_obj(self):
        """
        Returns: GPTQTrainer object
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_gptq_trainer method.')

    @abstractmethod
    def get_sensitivity_evaluation_fn(self,
                                      graph: Graph,
                                      quant_config: MixedPrecisionQuantizationConfig,
                                      metrics_weights: np.ndarray,
                                      representative_data_gen: Callable,
                                      fw_info: FrameworkInfo) -> Callable:
        """
        Create and return a function to compute a sensitivity metric for a mixed-precision
        configuration (comparing to the float model).

        Args:
            graph: Graph to build it's float and mixed-precision models.
            quant_config: QuantizationConfig of how the model should be quantized.
            metrics_weights: Array of weights to weight the sensitivity among different layers.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            A function that computes the metric.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_sensitivity_evaluation_fn method.')

    def get_node_prior_info(self, node: BaseNode,
                            fw_info: FrameworkInfo,
                            graph: Graph) -> NodePriorInfo:
        """
        Get a NodePriorInfo object for a node.

        Args:
            node: Node to get its prior info.
            fw_info: Framework specific information needed to create the prior info of the node.
            graph: Graph to check the next node type.

        Returns:
            NodePriorInfo with information about the node.
        """

        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s get_node_prior_info method.')
