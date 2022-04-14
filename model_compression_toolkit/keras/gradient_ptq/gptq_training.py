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
from typing import Callable, List, Tuple
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tqdm import tqdm

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if tf.__version__ < "2.6":
    from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
else:
    from keras.engine.base_layer import TensorFlowOpLayer

from model_compression_toolkit import common
from model_compression_toolkit.common.gptq.gptq_training import GPTQTrainer
from model_compression_toolkit.common.gptq.gptq_config import GradientPTQConfig
from model_compression_toolkit.common import Graph
from model_compression_toolkit.keras.gradient_ptq.graph_info import get_trainable_parameters, get_weights_for_loss
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.framework_implementation import FrameworkImplementation
import numpy as np
import copy
from model_compression_toolkit.keras.constants import BIAS, USE_BIAS
from model_compression_toolkit.keras.quantizer.gradient_ptq import WeightQuantizeConfig


class KerasGPTQTrainer(GPTQTrainer):
    """
    Keras GPTQ training class for fine-tuning a quantized model
    """
    def __init__(self,
                 graph_float: Graph,
                 graph_quant: Graph,
                 gptq_config: GradientPTQConfig,
                 fw_impl: FrameworkImplementation,
                 fw_info: FrameworkInfo):
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
            fw_info: Framework information
        """
        super().__init__(graph_float, graph_quant, gptq_config, fw_impl, fw_info)
        self.input_scale = 1
        trainable_weights = get_trainable_parameters(self.fxp_model,
                                                     fw_info,
                                                     add_bias=gptq_config.train_bias)

        self.flp_weights_list, self.fxp_weights_list = get_weights_for_loss(self.fxp_model)

        if not (len(self.compare_points) == len(trainable_weights) == len(self.flp_weights_list) == len(self.fxp_weights_list)):
            raise Exception(
                "GPTQ: Mismatch between number of compare points, number of layers with trainable weights " +
                "and number of float and quantized weights for loss")

        self.flattened_trainable_weights = [w for layer_weights in trainable_weights for w in layer_weights]

        if self.float_user_info.input_scale != self.gptq_user_info.input_scale:
            common.Logger.error("Input scale mismatch between float and GPTQ networks")  # pragma: no cover
        else:
            self.input_scale = self.gptq_user_info.input_scale

    def train(self, representative_data_gen: Callable):
        """
        Train the quantized model using GPTQ training process in Keras framework
        Args:
            representative_data_gen: Dataset to use for inputs of the models.
        """

        def update_step(input_data: List[np.ndarray]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
            """
            Get outputs from both teacher and student networks. Compute the observed error,
            and use it to compute the gradients and applying them to the student weights.
            Args:
                input_data: A list of Input tensors to pass through the networks.
            Returns:
                Loss and gradients.
            """
            y_float = self.float_model(input_data)  # running float model
            with tf.GradientTape(persistent=True) as tape:
                y_fxp = self.fxp_model(input_data)  # running fxp model
                loss_value = self.gptq_config.loss(y_fxp, y_float, self.fxp_weights_list, self.flp_weights_list,
                                              self.compare_points_mean, self.compare_points_std)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, self.flattened_trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.gptq_config.optimizer.apply_gradients(zip(grads, self.flattened_trainable_weights))

            return loss_value, grads

        # ----------------------------------------------
        # Training loop
        # ----------------------------------------------
        self.loss_list = []
        for _ in tqdm(range(self.gptq_config.n_iter)):
            data = representative_data_gen()
            loss_value_step, grads = update_step([d * self.input_scale for d in data])
            if self.gptq_config.log_function is not None:
                self.gptq_config.log_function(loss_value_step, grads, self.flattened_trainable_weights, self.compare_points)
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
