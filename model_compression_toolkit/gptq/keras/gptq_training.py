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

import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tqdm import tqdm

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.gptq.keras.gptq_model_builder import GPTQKerasModelBuilder

if tf.__version__ < "2.6":
    from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
else:
    from keras.engine.base_layer import TensorFlowOpLayer

from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.common.gptq_training import GPTQTrainer
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.gptq.keras.graph_info import get_trainable_parameters, get_weights_for_loss, \
    get_gumbel_probability
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
import numpy as np
import copy
from model_compression_toolkit.core.keras.constants import BIAS, USE_BIAS
from model_compression_toolkit.gptq.keras.quantizer import WeightQuantizeConfig
from model_compression_toolkit.gptq.keras.optimizers.sam_optimizer import SAM


class KerasGPTQTrainer(GPTQTrainer):
    """
    Keras GPTQ training class for fine-tuning a quantized model
    """

    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
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
            gptq_config: GradientPTQConfig with parameters about the tuning process.
            fw_impl: FrameworkImplementation object with a specific framework methods implementation.
            fw_info: Framework information.
            representative_data_gen: Dataset to use for inputs of the models.
        """
        super().__init__(graph_float,
                         graph_quant,
                         gptq_config,
                         fw_impl,
                         fw_info)

        self.loss_list = []
        self.input_scale = 1

        trainable_weights, bias_weights, trainable_threshold, temperature_weights = get_trainable_parameters(
            self.fxp_model,
            fw_info,
            add_bias=True,
            is_gumbel=gptq_config.is_gumbel)

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
                                                                  trainable_quantization_parameters,
                                                                  temperature_weights)
        self.has_params_to_train = np.sum([len(optimizer_params_tuple[1]) for optimizer_params_tuple in self.optimizer_with_param])>0

        if self.float_user_info.input_scale != self.gptq_user_info.input_scale:
            common.Logger.error("Input scale mismatch between float and GPTQ networks")  # pragma: no cover
        else:
            self.input_scale = self.gptq_user_info.input_scale

        self.weights_for_average_loss = self.compute_jacobian_based_weights(representative_data_gen)

    def build_gptq_model(self):
        """
        Build the GPTQ model with QuantizationWrappers
        Returns:
            Quantized graph for GPTQ fine-tuning, GPTQ graph user info
        """

        return GPTQKerasModelBuilder(graph=self.graph_quant,
                                     gptq_config=self.gptq_config,
                                     append2output=self.compare_points,
                                     fw_info=self.fw_info,
                                     return_float_outputs=True).build_model()

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

            if self.gptq_config.is_gumbel and self.gptq_config.quantizer_config.temperature_learning:
                gumbel_prob = get_gumbel_probability(self.fxp_model)
                gumbel_reg = 0
                for p in gumbel_prob:
                    entropy = -tf.reduce_mean(
                        tf.reduce_sum(p * tf.math.log(tf.maximum(p,
                                                                 self.gptq_config.eps)),
                                      axis=0))

                    gumbel_reg += entropy
                gumbel_reg /= len(gumbel_prob)
                loss_value += self.gptq_config.quantizer_config.gumbel_entropy_regularization * gumbel_reg

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
        if self.gptq_config.sam_optimization:
            sam = SAM(self.fxp_model, self.compute_gradients, self.optimizer_with_param, self.gptq_config.rho)
            compute_gradients = sam.compute_gradients

        # ----------------------------------------------
        # Training loop
        # ----------------------------------------------
        if self.has_params_to_train:
            self.micro_training_loop(representative_data_gen,
                                     compute_gradients,
                                     self.optimizer_with_param,
                                     self.gptq_config.n_iter,
                                     True)

    def micro_training_loop(self,
                            data_function: Callable,
                            in_compute_gradients: Callable,
                            in_optimizer_with_param: List[Tuple[tf.keras.optimizers.Optimizer, List[tf.Tensor]]],
                            n_iteration: int,
                            is_training: bool):
        """
        This function run a micro training loop on given set of parameters.
        Args:
            data_function: A callable function that give a batch of samples.
            in_compute_gradients: A callable function that compute the gradients.
            in_optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            n_iteration: Number of update iteration.
            is_training: A boolean flag stating if the network is running in training mode.

        Returns: None

        """
        for _ in tqdm(range(int(n_iteration))):
            data = data_function()
            input_data = [d * self.input_scale for d in data]
            y_float = self.float_model(input_data)  # running float model
            loss_value_step, grads = in_compute_gradients(y_float, input_data, in_optimizer_with_param,
                                                          training=is_training)
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            for i, (o, p) in enumerate(in_optimizer_with_param):
                o.apply_gradients(zip(grads[i], p))
            if self.gptq_config.log_function is not None:
                self.gptq_config.log_function(loss_value_step, grads[0], in_optimizer_with_param[0][-1],
                                              self.compare_points)
            self.loss_list.append(loss_value_step.numpy())
            common.Logger.debug(f'last loss value: {self.loss_list[-1]}')

    def update_graph(self):
        """
        Update a graph using GPTQ after minimizing the loss between the float model's output
        and the quantized model's outputs.
        Returns:
            Updated graph after GPTQ.
        """
        graph = copy.copy(self.graph_quant)

        for layer in self.fxp_model.layers:
            if isinstance(layer, QuantizeWrapper) and isinstance(
                    layer.quantize_config, WeightQuantizeConfig):
                node = graph.find_node_by_name(layer.layer.name)
                if len(node) == 0 and isinstance(layer.layer, TensorFlowOpLayer):
                    node = graph.find_node_by_name('_'.join(layer.layer.name.split('_')[3:]))
                if len(node) != 1:
                    common.Logger.error(f"Can't update GPTQ graph due to missing layer named: {layer.layer.name}")
                node = node[0]
                weights, weight_quant_config, activation_quant_config = \
                    layer.quantize_config.update_layer_quantization_params(layer)
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

