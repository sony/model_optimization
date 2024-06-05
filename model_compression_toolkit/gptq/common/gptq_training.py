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
import copy
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, List, Any, Dict

from model_compression_toolkit.constants import ACT_HESSIAN_DEFAULT_BATCH_SIZE
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR
from model_compression_toolkit.gptq.common.gptq_framework_implementation import GPTQFrameworkImplemantation
from model_compression_toolkit.gptq.common.gptq_graph import get_compare_points
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.hessian import HessianInfoService, TraceHessianRequest, HessianMode, \
    HessianInfoGranularity
from model_compression_toolkit.core.common.hessian import hessian_info_utils as hessian_utils


class GPTQTrainer(ABC):
    """
    Abstract GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: GPTQFrameworkImplemantation,
                 fw_info: FrameworkInfo,
                 hessian_info_service: HessianInfoService = None):
        """
        Build two models from a graph: A teacher network (float model) and a student network (quantized model).
        Use the dataset generator to pass images through the teacher and student networks to get intermediate
        layers outputs. Use the outputs to compute the observed loss and to back-propagate the error
        in the student network, to minimize it in the next similar steps.
        All parameters (such as number of iterations, optimizer, etc.) are in GradientPTQConfig.
        Args:
            graph_float: Graph to build a float networks from.
            graph_quant: Graph to build a quantized networks from.
            gptq_config: GradientPTQConfig with parameters about the tuning process.
            fw_impl: Framework implementation
            fw_info: Framework information
            hessian_info_service: HessianInfoService for fetching and computing Hessian's trace approximation.
        """
        self.graph_float = copy.deepcopy(graph_float)
        self.graph_quant = copy.deepcopy(graph_quant)
        self.gptq_config = gptq_config
        self.fw_impl = fw_impl
        self.fw_info = fw_info

        # ----------------------------------------------
        # Build two models and create compare nodes
        # ----------------------------------------------
        self.compare_points, _, self.compare_points_mean, self.compare_points_std = get_compare_points(self.graph_float)

        self.float_model, self.float_user_info = fw_impl.model_builder(self.graph_float,
                                                                       mode=ModelBuilderMode.FLOAT,
                                                                       append2output=self.compare_points,
                                                                       fw_info=self.fw_info)

        self.fxp_model, self.gptq_user_info = self.build_gptq_model()
        if self.gptq_config.use_hessian_based_weights:
            if not isinstance(hessian_info_service, HessianInfoService):
                Logger.critical(f"When using Hessian-based approximations for sensitivity evaluation, "
                                f"an 'HessianInfoService' object must be provided, but received: {hessian_info_service}.")   # pragma: no cover
            self.hessian_service = hessian_info_service

    def get_optimizer_with_param(self,
                                 flattened_trainable_weights: List[Any],
                                 flattened_bias_weights: List[Any],
                                 trainable_quantization_parameters: List[Any]) -> List[Any]:
        """
        Create Optimizers with their trainable parameters
        Args:
            flattened_trainable_weights: list of trainable weights parameters (flattened)
            flattened_bias_weights: list of trainable bias parameters (flattened)
            trainable_quantization_parameters: list of trainable quantization parameters
        Returns:
            List of Optimizer objects with parameters
        """

        w2train = [*flattened_trainable_weights]

        quant_params_learning = self.gptq_config.gptq_quantizer_params_override.get(QUANT_PARAM_LEARNING_STR, False)

        optimizer_with_param = [(self.gptq_config.optimizer, w2train)]
        if self.gptq_config.train_bias or quant_params_learning:
            w2train_res = []
            if self.gptq_config.train_bias:
                if self.gptq_config.optimizer_bias is not None:
                    optimizer_with_param.append((self.gptq_config.optimizer_bias, flattened_bias_weights))
                else:
                    w2train_res.extend(flattened_bias_weights)
                    if self.gptq_config.optimizer_rest is None:
                        Logger.critical("To enable bias micro-training, an additional optimizer is required. "
                                        "Please define the 'optimizer_rest' parameter.")# pragma: no cover
            if quant_params_learning:
                if self.gptq_config.optimizer_quantization_parameter is not None:  # Ability to override optimizer
                    optimizer_with_param.append((self.gptq_config.optimizer_quantization_parameter,
                                                 trainable_quantization_parameters))
                else:
                    w2train_res.extend(trainable_quantization_parameters)
                if self.gptq_config.optimizer_rest is None:
                    Logger.critical(
                        "To enable quantization parameters micro-training, an additional optimizer is required. "
                        "Please define the 'optimizer_rest' parameter.")  # pragma: no cover
            if len(w2train_res) > 0:
                # Either bias or quantization parameters are trainable but did not provide a specific optimizer,
                # so we should use optimizer_rest to train them
                if self.gptq_config.optimizer_rest is None:
                    Logger.critical(
                        "To enable bais or quantization parameters micro-training, an additional optimizer is required. "
                        "Please define the 'optimizer_rest' parameter.")  # pragma: no cover
                optimizer_with_param.append((self.gptq_config.optimizer_rest, w2train_res))

        return optimizer_with_param

    def compute_hessian_based_weights(self) -> np.ndarray:
        """
        Computes trace hessian approximations per layer w.r.t activations of the interest points.

        Returns:
            np.ndarray: Trace hessian approximations.
        """
        if not self.gptq_config.use_hessian_based_weights:
            # Return a default weight distribution based on the number of compare points
            num_nodes = len(self.compare_points)
            return np.asarray([1 / num_nodes for _ in range(num_nodes)])

        # Fetch hessian approximations for each target node
        compare_point_to_trace_hessian_approximations = self._fetch_hessian_approximations()
        # Process the fetched hessian approximations to gather them per images
        trace_hessian_approx_by_image = (
            self._process_hessian_approximations(compare_point_to_trace_hessian_approximations))

        # Check if log normalization is enabled in the configuration
        if self.gptq_config.hessian_weights_config.log_norm:
            # Calculate the mean of the approximations across images
            mean_approx_scores = np.mean(trace_hessian_approx_by_image, axis=0)
            # Reduce unnecessary dims, should remain with one dimension for the number of nodes
            mean_approx_scores = np.squeeze(mean_approx_scores)
            # Handle zero values to avoid log(0)
            mean_approx_scores = np.where(mean_approx_scores != 0, mean_approx_scores,
                                          np.partition(mean_approx_scores, 1)[1])

            # Calculate log weights
            log_weights = np.log10(mean_approx_scores)

            # Check if scaling of log normalization is enabled in the configuration
            if self.gptq_config.hessian_weights_config.scale_log_norm:
                # Scale the log weights to the range [0, 1]
                return (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))

            # Offset the log weights so the minimum value is 0
            return log_weights - np.min(log_weights)
        else:
            # If log normalization is not enabled, return the mean of the approximations across images
            return np.mean(trace_hessian_approx_by_image, axis=0)

    def _fetch_hessian_approximations(self) -> Dict[BaseNode, List[List[float]]]:
        """
        Fetches hessian approximations for each target node.

        Returns:
            Mapping of target nodes to their hessian approximations.
        """
        approximations = {}
        trace_hessian_request = TraceHessianRequest(
            mode=HessianMode.ACTIVATION,
            granularity=HessianInfoGranularity.PER_TENSOR,
            target_nodes=self.compare_points
        )
        node_approximations = self.hessian_service.fetch_hessian(
            trace_hessian_request=trace_hessian_request,
            required_size=self.gptq_config.hessian_weights_config.hessians_num_samples,
            batch_size=self.gptq_config.hessian_weights_config.hessian_batch_size
        )

        for i, target_node in enumerate(self.compare_points):
            approximations[target_node] = node_approximations[i]

        return approximations

    def _process_hessian_approximations(self, approximations: Dict[BaseNode, List[List[float]]]) -> List:
        """
        Processes the fetched hessian approximations by image.
        Receives a dictionary of Node to a list of the length of the number of images that were fetched.
        Returns list of lists where each inner list is the approximations per image to all interest points.

        Args:
            approximations: Hessian trace approximations mapping to process.
            Dictionary of Node to a list of the length of the number of images that were fetched.

        Returns:
            Processed approximations as a list of lists where each inner list is the approximations
             per image to all interest points.
        """
        trace_hessian_approx_by_image = [[approximations[target_node][image_idx] for target_node in self.compare_points]
                                         for image_idx in
                                         range(self.gptq_config.hessian_weights_config.hessians_num_samples)]

        if self.gptq_config.hessian_weights_config.norm_scores:
            trace_hessian_approx_by_image = hessian_utils.normalize_scores(trace_hessian_approx_by_image)

        return trace_hessian_approx_by_image

    def _get_approximations_by_interest_point(self, approximations: Dict, image_idx: int) -> List:
        """
        Retrieves hessian approximations for a specific image index.

        Args:
            approximations (Dict): Hessian approximations.
            image_idx (int): Image index.

        Returns:
            List: Hessian approximations for the given image index.
        """
        approx_by_interest_point = []
        for target_node in self.compare_points:
            trace_approx = approximations[target_node][image_idx]
            self._validate_trace_approximation(trace_approx)
            approx_by_interest_point.append(trace_approx[0])
        return approx_by_interest_point

    @staticmethod
    def _validate_trace_approximation(trace_approx: List):
        """
        Validates the structure and length of the trace approximation.

        Args:
            trace_approx: Trace approximation to validate.
        """
        if not isinstance(trace_approx, list):
            Logger.critical(f"Trace approximation was expected to be a list but is of type: {type(trace_approx)}.")   # pragma: no cover
        if len(trace_approx) != 1:
            Logger.critical(f"Trace approximation was expected to have a length of 1 "
                            f"(for computations with granularity set to 'HessianInfoGranularity.PER_TENSOR') "
                            f"but has a length of {len(trace_approx)}."
            )   # pragma: no cover


    @abstractmethod
    def build_gptq_model(self):
        """
        Build the GPTQ model with QuantizationWrappers
        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s GPTQ model builder method.')  # pragma: no cover

    @abstractmethod
    def train(self, representative_data_gen: Callable):
        """
        Train the quantized model using GPTQ training process
        Args:
            representative_data_gen: Dataset to use for inputs of the models.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s train method.')  # pragma: no cover

    @abstractmethod
    def update_graph(self) -> Graph:
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.
        Returns:
            Updated graph after GPTQ.
        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement the '
                             f'framework\'s update_graph method.')  # pragma: no cover

def gptq_training(graph_float: Graph,
                  graph_quant: Graph,
                  gptq_config: GradientPTQConfig,
                  representative_data_gen: Callable,
                  fw_impl: GPTQFrameworkImplemantation,
                  fw_info: FrameworkInfo,
                  hessian_info_service: HessianInfoService=None) -> Graph:
    """
    GPTQ training process using knowledge distillation with a teacher network (float model) and a student network (quantized model).
    Args:
        graph_float: Graph to build a float networks from.
        graph_quant: Graph to build a quantized networks from.
        gptq_config: GradientPTQConfig with parameters about the tuning process.
        representative_data_gen: Dataset to use for inputs of the models.
        fw_impl: Framework implementation
        fw_info: Framework information
        hessian_info_service: HessianInfoService to fetch Hessian traces approximations.

    Returns:
        Quantized graph for export

    """
    # Get GPTQ object and initialize it
    gptq_trainer_obj = fw_impl.get_gptq_trainer_obj()

    gptq_trainer = gptq_trainer_obj(graph_float,
                                    graph_quant,
                                    gptq_config,
                                    fw_impl,
                                    fw_info,
                                    representative_data_gen,
                                    hessian_info_service=hessian_info_service)

    # Training process
    gptq_trainer.train(representative_data_gen)

    # Update graph
    graph_quant = gptq_trainer.update_graph()

    return graph_quant
