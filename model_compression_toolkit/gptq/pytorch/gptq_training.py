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
from typing import Callable, List, Tuple

import numpy as np
from tqdm import tqdm
import copy
import torch
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfigV2
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.pytorch.constants import BIAS, KERNEL
from model_compression_toolkit.gptq.pytorch.gptq_model_builder import GPTQPytorchModelBuilder
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, set_model, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.gptq_graph_info import get_trainable_parameters, get_weights_for_loss
from model_compression_toolkit.gptq.pytorch.quantizer.quantizer_wrapper import WeightQuantizerWrapper
from model_compression_toolkit.gptq.pytorch.gptq_graph_info import get_gumbel_probability


class PytorchGPTQTrainer(GPTQTrainer):
    """
    Pytorch GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfigV2,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo,
                 representative_data_gen: Callable):
        """
        Build two models from a graph: A teacher network (float model) and a student network (quantized model).
        Use the dataset generator to pass images through the teacher and student networks to get intermediate
        layers outputs. Use the outputs to compute the observed loss and to back-propagate the error
        in the student network, to minimize it in the next similar steps.
        All parameters (such as number of iterations, optimizer, etc.) are in GradientPTQConfig.
        Args:
            graph_float: Graph to build a float networks from.
            graph_quant: Graph to build a quantized networks from.
            gptq_config: GradientPTQConfigV2 with parameters about the tuning process.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            fw_info: Framework information
            representative_data_gen: Dataset to use for inputs of the models.
        """
        super().__init__(graph_float, graph_quant, gptq_config, fw_impl, fw_info)
        self.loss_list = []
        self.input_scale = 1
        if self.float_user_info.input_scale != self.gptq_user_info.input_scale:
            Logger.error("Input scale mismatch between float and GPTQ networks")  # pragma: no cover
        else:
            self.input_scale = self.gptq_user_info.input_scale

        trainable_weights, trainable_bias, trainable_threshold, trainable_temperature = get_trainable_parameters(
            self.fxp_model,
            add_bias=self.gptq_config.train_bias,
            quantization_parameters_learning=self.gptq_config.quantization_parameters_learning,
            is_gumbel=self.gptq_config.is_gumbel)

        self.flp_weights_list, self.fxp_weights_list = get_weights_for_loss(self.fxp_model)
        if not (len(self.compare_points) == len(trainable_weights) == len(self.flp_weights_list) == len(
                self.fxp_weights_list)):
            Logger.error(
                "GPTQ: Mismatch between number of compare points, number of layers with trainable weights " +
                "and number of float and quantized weights for loss")

        self.optimizer_with_param = self.get_optimizer_with_param(trainable_weights,
                                                                  trainable_bias,
                                                                  trainable_threshold,
                                                                  trainable_temperature)

        self.weights_for_average_loss = to_torch_tensor(self.compute_jacobian_based_weights(representative_data_gen))

    def build_gptq_model(self):
        """
        Build the GPTQ model with QuantizationWrappers
        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """
        return GPTQPytorchModelBuilder(self.graph_quant,
                                       self.gptq_config,
                                       append2output=self.compare_points,
                                       return_float_outputs=True).build_model()

    def train(self, representative_data_gen: Callable):
        """
          GPTQ Training using pytorch framework
          Args:
              representative_data_gen: Dataset generator to get images.
          Returns:
              Graph after GPTQ training
          """
        # Set Optimizers
        for (optimizer, params) in self.optimizer_with_param:
            optimizer.param_groups.clear()
            optimizer.add_param_group({'params': params})

        # Set models mode
        set_model(self.float_model, False)
        set_model(self.fxp_model, True)
        self._set_requires_grad()

        # ----------------------------------------------
        # Training loop
        # ----------------------------------------------
        self.micro_training_loop(representative_data_gen, self.gptq_config.n_epochs)

    def compute_gradients(self,
                          y_float: List[torch.Tensor],
                          input_tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Get outputs from both teacher and student networks. Compute the observed error,
        and use it to compute the gradients and applying them to the student weights.
        Args:
            y_float: A list of reference tensor from the floating point network.
            input_tensors: A list of Input tensors to pass through the networks.
        Returns:
            Loss and gradients.
        """

        # Forward-pass
        y_fxp = self.fxp_model(input_tensors)

        # Loss
        loss_value = self.gptq_config.loss(y_fxp,
                                           y_float,
                                           self.fxp_weights_list,
                                           self.flp_weights_list,
                                           self.compare_points_mean,
                                           self.compare_points_std,
                                           self.weights_for_average_loss)

        if self.gptq_config.is_gumbel and self.gptq_config.quantizer_config.temperature_learning:
            gumbel_prob = get_gumbel_probability(self.fxp_model)
            gumbel_reg = 0
            for p in gumbel_prob:
                entropy = -torch.mean(torch.sum(p * torch.log(torch.maximum(p, self.gptq_config.eps*torch.ones_like(p))),dim=0))
                gumbel_reg += entropy
            gumbel_reg = 0 if gumbel_reg == 0 else gumbel_reg/len(gumbel_prob)
            loss_value += self.gptq_config.quantizer_config.gumbel_entropy_regularization * gumbel_reg

        # Back-pass
        loss_value.backward()

        # Get gradients
        grads = []
        for param in self.fxp_model.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(torch_tensor_to_numpy(param.grad))

        return loss_value, grads

    def micro_training_loop(self,
                            data_function: Callable,
                            n_epochs: int):
        """
        This function run a micro training loop on given set of parameters.
        Args:
            data_function: A callable function that give a batch of samples.
            n_epochs: Number of update iterations of representative dataset.
        """
        for _ in tqdm(range(n_epochs)):
            for data in data_function():
                input_data = [d * self.input_scale for d in data]
                input_tensor = to_torch_tensor(input_data)
                y_float = self.float_model(input_tensor)  # running float model
                loss_value, grads = self.compute_gradients(y_float, input_tensor)
                # Run one step of gradient descent by updating the value of the variables to minimize the loss.
                for (optimizer, _) in self.optimizer_with_param:
                    optimizer.step()
                    optimizer.zero_grad()
                if self.gptq_config.log_function is not None:
                    self.gptq_config.log_function(loss_value.item(),
                                                  torch_tensor_to_numpy(grads),
                                                  torch_tensor_to_numpy(self.optimizer_with_param[0][-1]))
                self.loss_list.append(loss_value.item())
                Logger.debug(f'last loss value: {self.loss_list[-1]}')

    def update_graph(self) -> Graph:
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.
        Returns:
            Updated graph after GPTQ.
        """
        graph_quant = copy.copy(self.graph_quant)

        # Update graph after training
        for name, layer in self.fxp_model.named_modules():
            if isinstance(layer, WeightQuantizerWrapper):
                node = self.graph_quant.find_node_by_name(name)
                if len(node) != 1:
                    Logger.error(f"Can't update GPTQ graph due to missing layer named: {name}")
                node = node[0]
                # Weight
                node.set_weights_by_keys(KERNEL, self.fw_impl.to_numpy(layer.weight_quantizer(layer.float_weight, training=False)))
                # Weight quantization params
                if self.gptq_config.quantization_parameters_learning:
                    node.final_weights_quantization_cfg.set_weights_quantization_param(layer.weight_quantizer.get_weight_quant_params())
                # Bias
                if self.gptq_config.train_bias:
                    node.set_weights_by_keys(BIAS, self.fw_impl.to_numpy(getattr(layer.op, BIAS)))

        return graph_quant

    def _set_requires_grad(self):
        """
        Set require_grad flag for trainable parameters for GPTQ training
        """
        # Float and Fxp models: freeze all the parameters in the network
        for param in self.float_model.parameters():
            param.requires_grad = False
        for param in self.fxp_model.parameters():
            param.requires_grad = False

        # Fxp model: unfreeze only trainable parameters
        for layer in self.fxp_model.modules():
            if isinstance(layer, WeightQuantizerWrapper):
                for param in layer.weight_quantizer.get_trainable_params():
                    param.requires_grad = True
                if self.gptq_config.train_bias and hasattr(layer.op, BIAS):
                    bias = getattr(layer.op, BIAS)
                    bias.requires_grad = True
