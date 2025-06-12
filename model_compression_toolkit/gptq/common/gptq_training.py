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
from typing import Callable, List, Any, Iterable, Optional, Generator

import numpy as np

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.hessian import HessianInfoService, HessianScoresRequest, HessianMode, \
    HessianScoresGranularity, hessian_info_utils as hessian_utils
from model_compression_toolkit.core.common.model_builder_mode import ModelBuilderMode
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR
from model_compression_toolkit.gptq.common.gptq_framework_implementation import GPTQFrameworkImplemantation
from model_compression_toolkit.gptq.common.gptq_graph import get_compare_points
from model_compression_toolkit.gptq.common.gradual_activation_quantization import \
    get_gradual_activation_quantizer_wrapper_factory
from model_compression_toolkit.gptq.common.regularization_factory import get_regularization
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.trainable_infrastructure.common.util import get_total_grad_steps


class GPTQTrainer(ABC):
    """
    Abstract GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: GPTQFrameworkImplemantation,
                 representative_data_gen_fn: Callable[[], Generator],
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
            representative_data_gen_fn: factory for representative data generator.
            hessian_info_service: HessianInfoService for fetching and computing Hessian-approximation information.
        """
        self.graph_float = copy.deepcopy(graph_float)
        self.graph_quant = copy.deepcopy(graph_quant)
        self.gptq_config = gptq_config
        self.fw_impl = fw_impl
        self.representative_data_gen_fn = representative_data_gen_fn

        def _get_total_grad_steps():
            return get_total_grad_steps(representative_data_gen_fn) * gptq_config.n_epochs

        self.gradual_act_quantizer_wrapper_factory = get_gradual_activation_quantizer_wrapper_factory(gptq_config,
                                                                                                      _get_total_grad_steps,
                                                                                                      self.fw_linear_annealing_scheduler)

        # ----------------------------------------------
        # Build two models and create compare nodes
        # ----------------------------------------------
        self.compare_points, _, self.compare_points_mean, self.compare_points_std = get_compare_points(self.graph_float)

        self.float_model, self.float_user_info = fw_impl.model_builder(self.graph_float,
                                                                       mode=ModelBuilderMode.FLOAT,
                                                                       append2output=self.compare_points)

        self.fxp_model, self.gptq_user_info = self.build_gptq_model()
        if self.gptq_config.hessian_weights_config:
            if not isinstance(hessian_info_service, HessianInfoService):
                Logger.critical(f"When using Hessian-based approximations for sensitivity evaluation, "
                                f"an 'HessianInfoService' object must be provided, but received: {hessian_info_service}.")   # pragma: no cover
            self.hessian_service = hessian_info_service

        self.reg_func = get_regularization(self.gptq_config,
                                           _get_total_grad_steps,
                                           self.fw_soft_quantizer_regularization,
                                           self.fw_linear_annealing_scheduler)
        self.loss_list = []
        self.input_scale = 1
        if self.float_user_info.input_scale != self.gptq_user_info.input_scale:
            Logger.critical("Input scale mismatch between float and GPTQ networks. "
                            "Ensure both networks have matching input scales.")  # pragma: no cover
        else:
            self.input_scale = self.gptq_user_info.input_scale

        trainable_weights, trainable_bias, trainable_threshold = self.fw_get_gptq_trainable_parameters_fn(
            self.fxp_model,
            add_bias=self.gptq_config.train_bias)
        self.flp_weights_list, self.fxp_weights_list = self.fw_get_weights_for_loss_fn(self.fxp_model)

        if not (len(self.compare_points) == len(trainable_weights) == len(self.flp_weights_list) == len(
                self.fxp_weights_list)):
            Logger.critical("Mismatch in the number of comparison points, layers with trainable weights, "
                            "and the number of float and quantized weights for loss calculation. "
                            "Ensure all these elements align to proceed with GPTQ training.")

        # In Keras we need to flatten the weights first before attaching the optimizer
        if len(trainable_weights) > 0 and isinstance(trainable_weights[0], (list, tuple)):
            trainable_weights = [w for layer_weights in trainable_weights for w in layer_weights]
        if len(trainable_bias) > 0 and isinstance(trainable_bias[0], (list, tuple)):
            trainable_bias = [w for layer_weights in trainable_bias for w in layer_weights]

        self.optimizer_with_param = self.get_optimizer_with_param(trainable_weights,
                                                                  trainable_bias,
                                                                  trainable_threshold)
        hessian_cfg = self.gptq_config.hessian_weights_config

        self.has_params_to_train = np.sum(
            [len(optimizer_params_tuple[1]) for optimizer_params_tuple in self.optimizer_with_param]) > 0
        self.use_sample_layer_attention = hessian_cfg and hessian_cfg.per_sample

        if self.use_sample_layer_attention:
            # normalization is currently not supported, make sure the config reflects it.
            if hessian_cfg.norm_scores or hessian_cfg.log_norm or hessian_cfg.scale_log_norm:
                raise NotImplementedError()
            self.train_dataloader = self._prepare_train_dataloader_sla(representative_data_gen_fn)
        else:
            self.train_dataloader = self._prepare_train_dataloader_for_non_sla(representative_data_gen_fn)

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

    def compute_hessian_based_weights(self, data_loader: Iterable) -> np.ndarray:
        """
        Computes scores based on the hessian approximation per layer w.r.t activations of the interest points.

        Returns:
            np.ndarray: Scores based on the hessian matrix approximation.
        """
        request = self._build_hessian_request(
            HessianScoresGranularity.PER_TENSOR,
            data_loader=data_loader,
            n_samples=self.gptq_config.hessian_weights_config.hessians_num_samples
        )
        layers_hessians = self.hessian_service.fetch_hessian(request)

        hessian_approx_score_by_image = np.stack([layers_hessians[node.name] for node in self.compare_points], axis=1)
        assert hessian_approx_score_by_image.shape[0] == self.gptq_config.hessian_weights_config.hessians_num_samples

        if self.gptq_config.hessian_weights_config.norm_scores:
            hessian_approx_score_by_image = hessian_utils.normalize_scores(hessian_approx_score_by_image)

        # Calculate the mean of the approximations across images
        mean_approx_scores = np.mean(hessian_approx_score_by_image, axis=0)
        # assert len(mean_approx_scores.shape) == len(self.compare_points)

        if not self.gptq_config.hessian_weights_config.log_norm:
            return mean_approx_scores

        # Reduce unnecessary dims, should remain with one dimension for the number of nodes
        mean_approx_scores = np.squeeze(mean_approx_scores)
        # Handle zero values to avoid log(0)
        mean_approx_scores = np.where(mean_approx_scores != 0, mean_approx_scores,
                                      np.partition(mean_approx_scores, 1)[1])

        # Calculate log weights
        log_weights = np.log10(mean_approx_scores)

        if self.gptq_config.hessian_weights_config.scale_log_norm:
            # Scale the log weights to the range [0, 1]
            return (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))

        # Offset the log weights so the minimum value is 0
        return log_weights - np.min(log_weights)

    def _build_hessian_request(self, granularity: HessianScoresGranularity, data_loader: Iterable,
                               n_samples: Optional[int]) -> HessianScoresRequest:
        """
        Build hessian request for hessian service.

        Args:
            granularity: requested granularity.
            data_loader: data loader yielding samples to compute hessians on.
            n_samples: request number of samples.

        Returns:
            Hessian request.
        """
        return HessianScoresRequest(
            mode=HessianMode.ACTIVATION,
            granularity=granularity,
            target_nodes=self.compare_points,
            data_loader=data_loader,
            n_samples=n_samples
        )

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
    def train(self):
        """
        Train the quantized model using GPTQ training process.
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
                  hessian_info_service: HessianInfoService = None) -> Graph:
    """
    GPTQ training process using knowledge distillation with a teacher network (float model) and a student network (quantized model).
    Args:
        graph_float: Graph to build a float networks from.
        graph_quant: Graph to build a quantized networks from.
        gptq_config: GradientPTQConfig with parameters about the tuning process.
        representative_data_gen: Dataset to use for inputs of the models.
        fw_impl: Framework implementation
        hessian_info_service: HessianInfoService to fetch information based on the Hessian approximation.

    Returns:
        Quantized graph for export

    """
    # Get GPTQ object and initialize it
    gptq_trainer_obj = fw_impl.get_gptq_trainer_obj()

    gptq_trainer = gptq_trainer_obj(graph_float,
                                    graph_quant,
                                    gptq_config,
                                    fw_impl,
                                    representative_data_gen,
                                    hessian_info_service=hessian_info_service)

    # Training process
    gptq_trainer.train()

    # Update graph
    graph_quant = gptq_trainer.update_graph()

    return graph_quant
