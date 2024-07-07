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
from typing import Callable, List, Tuple, Union

import numpy as np
from torch.nn import Module
from tqdm import tqdm
import copy
import torch

from model_compression_toolkit.core.common.hessian import HessianInfoService
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common import Graph, BaseNode
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.pytorch.constants import BIAS
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, set_model, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.graph_info import get_gptq_trainable_parameters, \
    get_weights_for_loss
from model_compression_toolkit.gptq.pytorch.quantizer.quantization_builder import quantization_builder
from model_compression_toolkit.gptq.pytorch.quantizer.regularization_factory import get_regularization
from mct_quantizers import PytorchQuantizationWrapper, PytorchActivationQuantizationHolder


class PytorchGPTQTrainer(GPTQTrainer):
    """
    Pytorch GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo,
                 representative_data_gen: Callable,
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
            gptq_config: GradientPTQConfigV2 with parameters about the tuning process.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            fw_info: Framework information
            representative_data_gen: Dataset to use for inputs of the models.
            hessian_info_service: HessianInfoService to fetch info based on the hessian approximation of the float model.
        """
        super().__init__(graph_float,
                         graph_quant,
                         gptq_config,
                         fw_impl,
                         fw_info,
                         hessian_info_service=hessian_info_service)

        self.loss_list = []
        self.input_scale = 1
        if self.float_user_info.input_scale != self.gptq_user_info.input_scale:
            Logger.critical("Input scale mismatch between float and GPTQ networks. "
                            "Ensure both networks have matching input scales.")  # pragma: no cover
        else:
            self.input_scale = self.gptq_user_info.input_scale

        trainable_weights, trainable_bias, trainable_threshold = get_gptq_trainable_parameters(
            self.fxp_model,
            add_bias=self.gptq_config.train_bias)

        self.flp_weights_list, self.fxp_weights_list = get_weights_for_loss(self.fxp_model)
        if not (len(self.compare_points) == len(trainable_weights) == len(self.flp_weights_list) == len(
                self.fxp_weights_list)):
            Logger.critical("GPTQ: Number of comparison points, layers with trainable weights, "
                            "and float vs. quantized weights for loss calculation do not match. "
                            "Verify consistency across these parameters for successful GPTQ training.")

        self.optimizer_with_param = self.get_optimizer_with_param(trainable_weights,
                                                                  trainable_bias,
                                                                  trainable_threshold)

        self.weights_for_average_loss = to_torch_tensor(self.compute_hessian_based_weights())

        self.reg_func = get_regularization(self.gptq_config, representative_data_gen)

    def _is_gptq_weights_trainable(self,
                                   node: BaseNode) -> bool:
        """
        A function for deciding if a layer should be fine-tuned during GPTQ.

        Args:
            node (BaseNode): Node for quantization decision

        Returns:
            A boolean whether the layer is to be wrapped with a Quantization Wrapper.
        """

        kernel_attr = self.fw_info.get_kernel_op_attributes(node.type)[0]
        return kernel_attr is not None and node.is_weights_quantization_enabled(kernel_attr)

    def gptq_wrapper(self,
                     n: BaseNode,
                     layer: Module) -> Union[PytorchQuantizationWrapper, Module]:
        """
        A function which takes a computational graph node and a pytorch layer and perform the quantization wrapping.

        Args:
            n: A node of mct graph.
            layer: A pytorch layer

        Returns: Wrapped layer if the layer should be wrap, otherwise returns the layer as is.
        """

        if self._is_gptq_weights_trainable(n):
            # If we are here, then the node has a kernel attribute to quantize and training during GPTQ
            weights_quantizers, _ = quantization_builder(n,
                                                         self.gptq_config,
                                                         self.fw_info.get_kernel_op_attributes(n.type)[0])

            if len(weights_quantizers) > 0:
                return PytorchQuantizationWrapper(layer,
                                                  weights_quantizers=weights_quantizers)

        # TODO: need to check if in this case, if there are other weights attributes that are not trainable but are
        #  quantized, do we need to wrap them as well?
        return layer

    def get_activation_quantizer_holder(self, n: BaseNode) -> Callable:
        """
        Retrieve a PytorchActivationQuantizationHolder layer to use for activation quantization of a node.
        If the layer is not supposed to be wrapped with an activation quantizer - return None.
        Args:
            n: Node to attach a PytorchActivationQuantizationHolder to its output.
        Returns:
            A PytorchActivationQuantizationHolder module for the node's activation quantization.
        """
        _, activation_quantizers = quantization_builder(n, self.gptq_config)
        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node we no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return PytorchActivationQuantizationHolder(activation_quantizers[0])
        Logger.critical(f"'PytorchActivationQuantizationHolder' requires exactly one quantizer, "
                        f"but {len(activation_quantizers)} were found for node {n.name}. "
                        f"Ensure the node is configured with a single activation quantizer.")

    def build_gptq_model(self):
        """
        Build the GPTQ model with QuantizationWrappers
        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """
        gptq_model, gptq_user_info = PyTorchModelBuilder(graph=self.graph_quant,
                                                         append2output=self.compare_points,
                                                         fw_info=self.fw_info,
                                                         wrapper=self.gptq_wrapper,
                                                         return_float_outputs=True,
                                                         get_activation_quantizer_holder_fn=self.get_activation_quantizer_holder).build_model()

        return gptq_model, gptq_user_info

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

        reg_value = self.reg_func(self.fxp_model, self.gptq_config.regularization_factor)

        loss_value += reg_value

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
        with tqdm(range(n_epochs), "Running GPTQ optimization") as epochs_pbar:
            for _ in epochs_pbar:
                with tqdm(data_function(), position=1, leave=False) as data_pbar:
                    for data in data_pbar:
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
            if isinstance(layer, PytorchQuantizationWrapper):
                node = self.graph_quant.find_node_by_name(name)
                if len(node) != 1:
                    Logger.critical(f"Cannot update GPTQ graph: Layer with name '{name}' is missing or not unique. "
                                    f"Ensure each layer has a unique name and exists within the graph for updates.")
                node = node[0]
                kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=node.type,
                                                                      fw_info=self.fw_info)
                # TODO: only kernel attributes are currently trained in GPTQ, so only the kernel weights need to be updated.
                #  To enable GPTQ for other attributes, this code needs to be modified.
                weights, weight_quant_config, activation_quant_config = \
                    layer.weights_quantizers[kernel_attribute].update_layer_quantization_params(layer)
                for weight_attr, weight in weights.items():
                    node.set_weights_by_keys(weight_attr, self.fw_impl.to_numpy(weight))
                for config_parameter_name, config_parameter_value in weight_quant_config.items():
                    node.final_weights_quantization_cfg.set_quant_config_attr(config_parameter_name,
                                                                              config_parameter_value,
                                                                              attr_name=kernel_attribute)
                for config_attr, config_value in activation_quant_config.items():
                    node.final_activation_quantization_cfg.set_quant_config_attr(config_attr, config_value)
                if self.gptq_config.train_bias and hasattr(layer.layer, BIAS):
                    bias = getattr(layer.layer, BIAS)
                    if bias is not None:
                        node.set_weights_by_keys(BIAS, self.fw_impl.to_numpy(bias))

        return graph_quant

    def _set_requires_grad(self):
        """
        Set require_grad flag for trainable parameters for GPTQ training
        """
        # Float model: freeze all the parameters in the network
        for param in self.float_model.parameters():
            param.requires_grad = False

        # Fxp model: unfreeze bias trainable parameters
        for layer in self.fxp_model.modules():
            if isinstance(layer, PytorchQuantizationWrapper):
                if hasattr(layer.layer, BIAS):
                    bias = getattr(layer.layer, BIAS)
                    if bias is not None:
                        bias.requires_grad = self.gptq_config.train_bias
