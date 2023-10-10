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

import tensorflow as tf
from keras import Model
from packaging import version
from tensorflow.keras.layers import Layer
from tqdm import tqdm

from model_compression_toolkit.core.common.hessian import HessianInfoService
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.gptq.keras.quantizer.quantization_builder import quantization_builder
from model_compression_toolkit.logger import Logger
from mct_quantizers import KerasActivationQuantizationHolder

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.base_layer import TensorFlowOpLayer
else:
    from keras.engine.base_layer import TensorFlowOpLayer

from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfigV2
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.gptq.keras.graph_info import get_weights_for_loss, get_gptq_trainable_parameters
from model_compression_toolkit.gptq.keras.quantizer.regularization_factory import get_regularization
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
import numpy as np
import copy
from model_compression_toolkit.core.keras.constants import BIAS, USE_BIAS


class KerasGPTQTrainer(GPTQTrainer):
    """
    Keras GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfigV2,
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
            gptq_config: GradientPTQConfig with parameters about the tuning process.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            fw_info: Framework information.
            representative_data_gen: Dataset to use for inputs of the models.
            hessian_info_service: HessianInfoService for fetching and computing Hessian's trace approximation.

        """
        super().__init__(graph_float,
                         graph_quant,
                         gptq_config,
                         fw_impl,
                         fw_info,
                         hessian_info_service=hessian_info_service)

        self.loss_list = []
        self.input_scale = 1

        trainable_weights, bias_weights, trainable_threshold = get_gptq_trainable_parameters(
            self.fxp_model,
            fw_info,
            add_bias=gptq_config.train_bias)

        self.flp_weights_list, self.fxp_weights_list = get_weights_for_loss(self.fxp_model)

        if not (len(self.compare_points) == len(trainable_weights) == len(self.flp_weights_list) == len(
                self.fxp_weights_list)):
            raise Exception(
                "GPTQ: Mismatch between number of compare points, number of layers with trainable weights " +
                "and number of float and quantized weights for loss")

        flattened_trainable_weights = [w for layer_weights in trainable_weights for w in layer_weights]
        flattened_bias_weights = [w for layer_weights in bias_weights for w in layer_weights]
        trainable_quantization_parameters = trainable_threshold
        self.optimizer_with_param = self.get_optimizer_with_param(flattened_trainable_weights,
                                                                  flattened_bias_weights,
                                                                  trainable_quantization_parameters)
        self.has_params_to_train = np.sum(
            [len(optimizer_params_tuple[1]) for optimizer_params_tuple in self.optimizer_with_param]) > 0

        if self.float_user_info.input_scale != self.gptq_user_info.input_scale:
            Logger.error("Input scale mismatch between float and GPTQ networks")  # pragma: no cover
        else:
            self.input_scale = self.gptq_user_info.input_scale

        self.weights_for_average_loss = self.compute_hessian_based_weights()

        self.reg_func = get_regularization(self.gptq_config, representative_data_gen)

    def _is_gptq_weights_trainable(self,
                                   node: common.BaseNode) -> bool:
        """
        A function for deciding if a layer should be fine-tuned during GPTQ.

        Args:
            node (BaseNode): Node for quantization decision

        Returns:
            A boolean whether the layer is to be wrapped with a QuantizeWrapper
        """

        if node.is_weights_quantization_enabled() and not self.fw_info.is_kernel_op(node.type):
            Logger.error(f"GPTQ Error: Quantizing node {node.name} of type {node.type} "
                                f"without a kernel isn't supported")
        return node.is_weights_quantization_enabled()

    def gptq_wrapper(self,
                     n: common.BaseNode,
                     layer: Layer) -> Union[KerasTrainableQuantizationWrapper, Layer]:
        """
        A function which takes a computational graph node and a keras layer and perform the quantization wrapping.

        Args:
            n: A node of mct graph.
            layer: A keras layer

        Returns: Wrapped layer if the layer should be wrap, otherwise returns the layer as is.

        """
        if self._is_gptq_weights_trainable(n):
            weights_quantizers, _ = quantization_builder(n,
                                                         self.gptq_config) # TODO: split quantizers building into two functions: for weights and activations
            if len(weights_quantizers) > 0:
                return KerasTrainableQuantizationWrapper(layer,
                                                   weights_quantizers=weights_quantizers)
        return layer

    def get_activation_quantizer_holder(self, n: common.BaseNode) -> Callable:
        """
        Retrieve a KerasActivationQuantizationHolder layer to use for activation quantization for a node.
        If the layer is not supposed to be wrapped with activation quantizers - return None.

        Args:
            n: Node to get KerasActivationQuantizationHolder to attach in its output.

        Returns:
            A KerasActivationQuantizationHolder layer for the node activation quantization.
        """
        _, activation_quantizers = quantization_builder(n, self.gptq_config) # TODO: split quantizers building into two functions: for weights and activations

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return KerasActivationQuantizationHolder(activation_quantizers[0])

        Logger.error(
            f'KerasActivationQuantizationHolder supports a single quantizer but {len(activation_quantizers)} quantizers '
            f'were found for node {n}')


    def build_gptq_model(self) -> Tuple[Model, UserInformation]:
        """
        Build the GPTQ model with QuantizationWrappers

        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """

        gptq_model, gptq_user_info = KerasModelBuilder(graph=self.graph_quant,
                                                       append2output=self.compare_points,
                                                       fw_info=self.fw_info,
                                                       return_float_outputs=True,
                                                       wrapper=self.gptq_wrapper,
                                                       get_activation_quantizer_holder_fn=self.get_activation_quantizer_holder).build_model()

        return gptq_model, gptq_user_info

    def compute_gradients(self, in_y_float: List[tf.Tensor], input_data: List[np.ndarray],
                          in_optimizer_with_param: List,
                          training=True) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Get outputs from both teacher and student networks. Compute the observed error,
        and use it to compute the gradients and applying them to the student weights.
        Args:
            in_y_float: A list of reference tensor from the floating point network.
            input_data: A list of Input tensors to pass through the networks.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            training: A boolean flag stating if the network is running in training mode.

        Returns:
            Loss and gradients.
        """
        param2grad = []
        for _, p in in_optimizer_with_param:
            param2grad.extend(p)

        with tf.GradientTape(persistent=True) as tape:
            y_fxp = self.fxp_model(input_data, training=training)  # running fxp model
            loss_value = self.gptq_config.loss(y_fxp,
                                               in_y_float,
                                               self.fxp_weights_list,
                                               self.flp_weights_list,
                                               self.compare_points_mean,
                                               self.compare_points_std,
                                               self.weights_for_average_loss)

            reg_value = self.reg_func(self.fxp_model, self.gptq_config.regularization_factor)

            loss_value += reg_value

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, param2grad)
        res = []
        i = 0
        for _, p in in_optimizer_with_param:
            res.append(grads[i:(i + len(p))])
            i += len(p)
        return loss_value, res

    def train(self, representative_data_gen: Callable):
        """
        Train the quantized model using GPTQ training process in Keras framework
        Args:
            representative_data_gen: Dataset to use for inputs of the models.
        """
        compute_gradients = self.compute_gradients

        # ----------------------------------------------
        # Training loop
        # ----------------------------------------------
        if self.has_params_to_train:
            self.micro_training_loop(representative_data_gen,
                                     compute_gradients,
                                     self.optimizer_with_param,
                                     self.gptq_config.n_epochs,
                                     True)

    @tf.function
    def nano_training_step(self, input_data, in_compute_gradients, in_optimizer_with_param, is_training):
        """
        This function run part of the training step, wrapped by a tf.function for acceleration.
        Args:
            input_data: input data for the step.
            in_compute_gradients: A callable function that compute the gradients.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            is_training: A boolean flag stating if the network is running in training mode.

        Returns:
            loss value and gradients

        """

        # run float model
        y_float = self.float_model(input_data)
        # rung quantized model and calculate loss & gradients
        loss_value_step, grads = in_compute_gradients(y_float, input_data, in_optimizer_with_param,
                                                      training=is_training)
        return loss_value_step, grads

    def micro_training_loop(self,
                            data_function: Callable,
                            in_compute_gradients: Callable,
                            in_optimizer_with_param: List[Tuple[tf.keras.optimizers.Optimizer, List[tf.Tensor]]],
                            n_epochs: int,
                            is_training: bool):
        """
        This function run a micro training loop on given set of parameters.
        Args:
            data_function: A callable function that give a batch of samples.
            in_compute_gradients: A callable function that compute the gradients.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            n_epochs: Number of update iterations of representative dataset.
            is_training: A boolean flag stating if the network is running in training mode.

        Returns: None

        """
        for _ in tqdm(range(n_epochs)):
            for data in tqdm(data_function()):
                input_data = [d * self.input_scale for d in data]

                loss_value_step, grads = self.nano_training_step(input_data, in_compute_gradients,
                                                                 in_optimizer_with_param, is_training)
                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                for i, (o, p) in enumerate(in_optimizer_with_param):
                    o.apply_gradients(zip(grads[i], p))
                if self.gptq_config.log_function is not None:
                    self.gptq_config.log_function(loss_value_step, grads[0], in_optimizer_with_param[0][-1],
                                                  self.compare_points)
                self.loss_list.append(loss_value_step.numpy())
                Logger.debug(f'last loss value: {self.loss_list[-1]}')

    def update_graph(self):
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.
        Returns:
            Updated graph after GPTQ.
        """
        graph = copy.copy(self.graph_quant)

        for layer in self.fxp_model.layers:
            if isinstance(layer, KerasTrainableQuantizationWrapper):
                node = graph.find_node_by_name(layer.layer.name)
                if len(node) == 0 and isinstance(layer.layer, TensorFlowOpLayer):
                    node = graph.find_node_by_name('_'.join(layer.layer.name.split('_')[3:]))
                if len(node) != 1:
                    Logger.error(f"Can't update GPTQ graph due to missing layer named: {layer.layer.name}")
                node = node[0]
                kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=node.type,
                                                                      fw_info=self.fw_info)
                weights, weight_quant_config, activation_quant_config = \
                    layer.weights_quantizers[kernel_attribute].update_layer_quantization_params(layer)
                for weight_attr, weight in weights.items():
                    node.set_weights_by_keys(weight_attr, weight.numpy())
                for config_attr, config_value in weight_quant_config.items():
                    node.final_weights_quantization_cfg.set_quant_config_attr(config_attr, config_value)
                for config_attr, config_value in activation_quant_config.items():
                    node.final_activation_quantization_cfg.set_quant_config_attr(config_attr, config_value)
                if self.gptq_config.train_bias:
                    use_bias = layer.layer.get_config().get(USE_BIAS)
                    if use_bias is not None and use_bias:
                        new_bias = layer.layer.bias.numpy()
                        node.set_weights_by_keys(BIAS, new_bias)

        return graph
