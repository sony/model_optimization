# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Callable, Any, List, Tuple, Dict, Generator

import numpy as np

from model_compression_toolkit.constants import HESSIAN_NUM_ITERATIONS
from model_compression_toolkit.core import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.collectors.statistics_collector import BaseStatsCollector
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.base_graph import Graph
from model_compression_toolkit.core.common.hessian import HessianScoresRequest, HessianInfoService
from model_compression_toolkit.core.common.mixed_precision.sensitivity_evaluation import SensitivityEvaluation
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.core.common.node_prior_info import NodePriorInfo
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.user_info import UserInformation


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
        raise NotImplementedError(f'{self.__class__.__name__} did not supply a constants module.')  # pragma: no cover

    @abstractmethod
    def get_hessian_scores_calculator(self,
                                      graph: Graph,
                                      input_images: List[Any],
                                      hessian_scores_request: HessianScoresRequest,
                                      num_iterations_for_approximation: int = HESSIAN_NUM_ITERATIONS):
        """
        Get framework hessian-approximation scores calculator based on the hessian scores request.
        Args:
            input_images: Images to use for computation.
            graph: Float graph to compute the approximation of its different nodes.
            hessian_scores_request: HessianScoresRequest to search for the desired calculator.
            num_iterations_for_approximation: Number of iterations to use when approximating the Hessian-approximation scores.

        Returns: HessianScoresCalculator to use for the hessian approximation scores computation for this request.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_hessian_scores_calculator method.')  # pragma: no cover

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """
        Convert framework's tensor to a Numpy array.
        Args:
            tensor: Framework's tensor.

        Returns:
            Numpy array converted from the input tensor.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s to_numpy method.')  # pragma: no cover

    @abstractmethod
    def to_tensor(self, tensor: np.ndarray) -> Any:
        """
        Convert a Numpy array to a framework's tensor.
        Args:
            tensor: Numpy array.

        Returns:
            Framework's tensor converted from the input Numpy array.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s to_tensor method.')  # pragma: no cover

    @abstractmethod
    def is_tuple_of_tensors(self, obj: Any) -> bool:
        """
        Check if a given object if a tuple of tensors
        :param obj: Object to check its type
        :return: True if obj is a tuple of tensors, False otherwise
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s is_tuple_of_tensors method.')  # pragma: no cover


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
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s model_reader method.')  # pragma: no cover

    @abstractmethod
    def model_builder(self,
                      graph: Graph,
                      mode: ModelBuilderMode,
                      append2output: List[Any],
                      fw_info: FrameworkInfo,
                      return_float_outputs: bool = False) -> Tuple:
        """
        Build a framework model from a graph.
        The mode determines how the model should be build. append2output is a list of Nodes
        to set as the model outputs.

        Args:
            graph: Graph to build the model from it.
            mode: Mode for how to build the model.
            append2output: List of Nodes to set as the model's outputs.
            fw_info: FrameworkInfo object with information about the specific framework's model
            return_float_outputs (bool): whether to return outputs before or after quantization nodes (default)

        Returns:
            A tuple with the model and additional relevant supporting objects.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s model_builder method.')  # pragma: no cover

    @abstractmethod
    def run_model_inference(self,
                            model: Any,
                            input_list: List[Any],
                            requires_grad: bool = False) -> Tuple[Any]:
        """
        Executes the given model on the provided input data.

        This method must be implemented by subclasses to provide framework-specific logic
        for running inference (e.g., PyTorch, TensorFlow/Keras).

        Args:
            model: The framework-specific model instance.
            input_list: A list of inputs for the model.
            requires_grad: Whether to enable gradient computation. Defaults to `False`.

        Returns:
            The frameworks model's output.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s run_model_inference method.')  # pragma: no cover

    @abstractmethod
    def shift_negative_correction(self,
                                  graph: Graph,
                                  core_config: CoreConfig,
                                  fw_info: FrameworkInfo) -> Graph:
        """
        Apply shift negative correction (SNC) on a graph.

        Args:
            graph: Graph to apply SNC on.
            core_config: Quantization configuration.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Graph after SNC.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s apply_shift_negative_correction method.')  # pragma: no cover

    @abstractmethod
    def compute_activation_bias_correction(self,
                                           graph: Graph,
                                           quant_config: QuantizationConfig,
                                           fw_info: FrameworkInfo) -> Graph:
        """
        Compute activation bias correction on a graph.

        Args:
            graph: Graph to apply activation bias correction on.
            quant_config: QuantizationConfig of how the model should be quantized.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns:
            Graph after activation bias correction computing.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                                  f'framework\'s compute_activation_bias_correction method.')  # pragma: no cover

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
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_channel_equalization method.')  # pragma: no cover

    @abstractmethod
    def get_substitutions_prepare_graph(self, fw_info: FrameworkInfo = None) -> List[common.BaseSubstitution]:
        """

        Returns: A list of the framework substitutions used to prepare the graph.

        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_prepare_graph method.')  # pragma: no cover

    @abstractmethod
    def get_substitutions_pre_statistics_collection(self, quant_config: QuantizationConfig) -> \
            List[common.BaseSubstitution]:
        """

        Args:
            quant_config: Quantization configuration.

        Returns: A list of the framework substitutions used before we collect statistics.

        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_pre_statistics_collection method.')  # pragma: no cover

    @abstractmethod
    def get_linear_collapsing_substitution(self) -> common.BaseSubstitution:
        """
        Returns: linear collapsing substitution
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_linear_collapsing_substitution method.')  # pragma: no cover

    @abstractmethod
    def get_op2d_add_const_collapsing_substitution(self) -> common.BaseSubstitution:
        """
        Returns: conv2d add const collapsing substitution
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_op2d_add_const_collapsing_substitution method.')  # pragma: no cover

    @abstractmethod
    def get_substitutions_statistics_correction(self, quant_config: QuantizationConfig) -> \
            List[common.BaseSubstitution]:
        """
        Returns A list of the framework substitutions used for statistics correction.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used for statistics correction.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_statistics_correction method.')  # pragma: no cover

    @abstractmethod
    def get_residual_collapsing_substitution(self) -> List[common.BaseSubstitution]:
        """
        Returns: A list of the framework substitutions used for residual collapsing
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_residual_collapsing_substitution method.')  # pragma: no cover


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
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_post_statistics_collection method.')  # pragma: no cover

    @abstractmethod
    def get_substitutions_virtual_weights_activation_coupling(self) -> List[common.BaseSubstitution]:
        """
        Returns: A list of Keras substitutions used to build a virtual graph with composed activation-weights pairs.
        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_virtual_weights_activation_coupling '
                             f'method.')  # pragma: no cover

    @abstractmethod
    def get_substitutions_after_second_moment_correction(self, quant_config: QuantizationConfig) \
            -> List[common.BaseSubstitution]:
        """
        Return a list of the framework substitutions used after second moment statistics.

        Args:
            quant_config: QuantizationConfig to determine which substitutions to return.

        Returns:
            A list of the framework substitutions used after we apply second moment statistics.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_substitutions_after_second_moment_correction '
                             f'method.')  # pragma: no cover

    @abstractmethod
    def get_sensitivity_evaluator(self,
                                  graph: Graph,
                                  quant_config: MixedPrecisionQuantizationConfig,
                                  representative_data_gen: Callable,
                                  fw_info: FrameworkInfo,
                                  hessian_info_service: HessianInfoService = None,
                                  disable_activation_for_metric: bool = False) -> SensitivityEvaluation:
        """
        Creates and returns an object which handles the computation of a sensitivity metric for a mixed-precision
        configuration (comparing to the float model).

        Args:
            graph: Graph to build its float and mixed-precision models.
            quant_config: QuantizationConfig of how the model should be quantized.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            fw_info: FrameworkInfo object with information about the specific framework's model.
            disable_activation_for_metric: Whether to disable activation quantization when computing the MP metric.
            hessian_info_service: HessianInfoService to fetch information based on Hessian-approximation.

        Returns:
            A function that computes the metric.
        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_sensitivity_evaluator method.')  # pragma: no cover

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

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_node_prior_info method.')  # pragma: no cover

    def count_node_for_mixed_precision_interest_points(self, node: BaseNode) -> bool:
        """
        Returns whether a given node in considered as a potential interest point for mp metric computation purposes.
        Args:
            node: Node to indicate whether it needs to be part of the interest points set.
        Returns: True if the node should be considered an interest point, False otherwise.
        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s count_node_for_mixed_precision_interest_points method.')  # pragma: no cover

    def get_mp_node_distance_fn(self, n: BaseNode,
                                compute_distance_fn: Callable = None,
                                norm_mse: bool = False) -> Tuple[Callable, int]:
        """
        A mapping between layers' types and a distance function for computing the distance between
        two tensors in mixed precision (for loss computation purposes). Returns a specific function if node of specific types is
        given, or a default (normalized MSE) function otherwise.

        Args:
            n: Node to choose distance function for.
            compute_distance_fn: An optional distance function to use globally for all nodes.
            norm_mse: whether to normalize mse distance function.

        Returns: A distance function between two tensors and a axis on which the distance is computed (if exists).
        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_mp_node_distance_fn method.')  # pragma: no cover


    @abstractmethod
    def is_output_node_compatible_for_hessian_score_computation(self,
                                                                node: BaseNode) -> bool:
        """
        Checks and returns whether the given node is compatible as output for Hessian-based information computation.

        Args:
            node: A BaseNode object.

        Returns: Whether the node is compatible as output for Hessian-based information computation.

        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s is_output_node_compatible_for_hessian_score_computation method.')  # pragma: no cover

    @abstractmethod
    def get_node_mac_operations(self,
                                node: BaseNode,
                                fw_info: FrameworkInfo) -> float:
        """
        Gets the MAC operation count for a given operation.

        Args:
            node: A graph node that wraps the operation for which the MAC count is computed.
            fw_info: FrameworkInfo object with information about the specific framework's model.

        Returns: The MAC count of the operation
        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_node_mac_operations method.')  # pragma: no cover

    @abstractmethod
    def apply_second_moment_correction(self,
                                       quantized_model: Any,
                                       core_config: CoreConfig,
                                       representative_data_gen: Callable,
                                       graph: common.Graph):
        """
        Build a framework model from a graph and apply second moment statistics correction to graph.

        Args:
            quantized_model: Framework's model to apply second moment correction on.
            core_config: QuantizationConfig of how the model should be quantized.
            representative_data_gen: Dataset to use for retrieving images for the models inputs.
            graph: Graph to update the parameters after the second moment correction.

        Returns:
            A Graph after second moment correction.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s apply_second_moment_correction method.')  # pragma: no cover

    @abstractmethod
    def sensitivity_eval_inference(self,
                                   model: Any,
                                   inputs: Any):
        """
        Calls for a model inference for a specific framework during mixed precision sensitivity evaluation.

        Args:
            model: A model to run inference for.
            inputs: Input tensors to run inference on.

        Returns:
            The output of the model inference on the given input.
        """
        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s sensitivity_eval_inference method.')  # pragma: no cover

    def get_inferable_quantizers(self, node: BaseNode):
        """
        Returns sets of framework compatible weights and activation quantizers for the given node.

        Args:
           node: Node to get quantizers for.

        Returns:
            weight_quantizers: A dictionary between a weight's name to its quantizer.
            activation_quantizers: A list of activations quantization, one for each layer output.

        """

        raise NotImplementedError(f'{self.__class__.__name__} has to implement the '
                             f'framework\'s get_inferable_quantizers method.')  # pragma: no cover

    @staticmethod
    def convert_data_gen_to_dataloader(data_gen_fn: Callable[[], Generator], batch_size: int):
        """
        Create DataLoader based on samples yielded by data_gen.

        Args:
            data_gen_fn: data generator factory.
            batch_size: target batch size.

        Returns:
            Framework dataloader.
        """
        raise NotImplementedError()    # pragma: no cover
