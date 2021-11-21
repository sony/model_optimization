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

from typing import Callable, List, Tuple

import tensorflow as tf
from tqdm import tqdm

from model_compression_toolkit import common
from model_compression_toolkit.common import Graph
from model_compression_toolkit.keras.back2framework.model_builder import model_builder, ModelBuilderMode
from model_compression_toolkit.keras.gradient_ptq.graph_info import get_compare_points, \
    get_trainable_parameters
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.keras.gradient_ptq.graph_update import update_graph_after_gptq
from model_compression_toolkit.keras.gradient_ptq.gptq_loss import \
    multiple_tensors_mse_loss
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
import numpy as np


class GradientPTQConfig:
    """
    Configuration to use for quantization with GradientPTQ (experimental).
    """

    def __init__(self,
                 n_iter: int,
                 optimizer: OptimizerV2 = tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss: Callable = multiple_tensors_mse_loss,
                 log_function: Callable = None,
                 train_bias: bool = True,
                 representative_data_gen: Callable = None):
        """
        Initialize a GradientPTQConfig.

        Args:
            n_iter (int): Number of iterations to train.
            optimizer (OptimizerV2): Optimizer to use.
            loss (Callable): the loss to use. should accept 2 lists of tf.Tensor. 1st list are the quantized tensors, the 2nd the float tensors
            log_function (Callable): Function to log information about the GPTQ process.
            train_bias (bool): Whether to update the bias during the training or not.
            representative_data_gen (Callable): Dataset generator.

        Examples:
            Create a GradientPTQConfig to run for 5 iteration and uses a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]
            >>> gptq_conf = GradientPTQConfig(n_iter=5, representative_data_gen=repr_datagen)

            An optimizer can be passed:

            >>> gptq_conf = GradientPTQConfig(n_iter=5, representative_data_gen=repr_datagen, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.2))

            To disable the biases training, one may set train_bias to False (enabled by default):

            >>> gptq_conf = GradientPTQConfig(n_iter=5, representative_data_gen=repr_datagen, train_bias=False)

            The configuration can then be passed to :func:`~model_compression_toolkit.keras_post_training_quantization`.

        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.loss = loss
        self.log_function = log_function
        self.train_bias = train_bias
        self.representative_data_gen = representative_data_gen


def gptq_training_wrapper(tg: Graph,
                          representative_data_gen: Callable,
                          gptq_config: GradientPTQConfig,
                          fw_info: FrameworkInfo):
    """
    Build two models from a graph: A teacher network (float model) and a student network (quantized model).
    Use the dataset generator to pass images through the teacher and student networks to get intermediate
    layers outputs. Use the outputs to compute the observed loss and to backpropagate the error
    in the student network, to minimize it in the next similar steps.
    All parameters (such as number of iterations, optimizer, etc.) are in the passed GradientPTQConfig.

    Args:
        tg: Graph to build networks from.
        representative_data_gen: Dataset generator to get images.
        gptq_config: GradientPTQConfig with parameters about the tuning process.
        fw_info: Framework information needed for keras kernel ops list.

    Returns:
        Graph of quantized model after GPTQ training.
    """

    input_scale = 1
    gptq_representative_data_gen = gptq_config.representative_data_gen if gptq_config.representative_data_gen is not None \
        else representative_data_gen
    #########################################
    # Build two models and compare points
    #########################################
    compare_points, _ = get_compare_points(tg)  # get compare points
    n = len(compare_points)
    float_model, float_user_info = model_builder(tg,
                                                 mode=ModelBuilderMode.FLOAT,
                                                 append2output=compare_points)
    fxp_model, gptq_user_info = model_builder(tg,
                                            mode=ModelBuilderMode.GPTQ,
                                            append2output=compare_points)

    trainable_weights = get_trainable_parameters(fxp_model,
                                                 fw_info,
                                                 add_bias=gptq_config.train_bias)

    if float_user_info.input_scale != gptq_user_info.input_scale:
        common.Logger.error("Input scale mismatch between float and GPTQ networks")  # pragma: no cover
    else:
        input_scale = gptq_user_info.input_scale
    #########################################
    # Optimization Loop
    #########################################

    def update_step(input_data: List[np.ndarray]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Get outputs from both teacher and student networks. Compute the observed error,
        and use it to compute the gradients and applying them to the student weights.
        Args:
            input_data: A list of Input tensors to pass through the networks.

        Returns:
            Loss and gradients.
        """
        y_float = float_model(input_data)  # running float model
        with tf.GradientTape(persistent=True) as tape:
            y_fxp = fxp_model(input_data)  # running fxp model
            loss_value = gptq_config.loss(y_fxp, y_float)  # calculate cs loss

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        gptq_config.optimizer.apply_gradients(zip(grads, trainable_weights))

        return loss_value, grads

    loss_list = []
    for _ in tqdm(range(gptq_config.n_iter)):
        data = gptq_representative_data_gen()
        loss_value_step, grads = update_step([d * input_scale for d in data])
        if gptq_config.log_function is not None:
            gptq_config.log_function(loss_value_step, grads, trainable_weights, compare_points)
        loss_list.append(loss_value_step.numpy())
        common.Logger.debug(f'last loss value: {loss_list[-1]}')

    #########################################
    # Update Graph after GPTQ
    #########################################
    tg_gptq = update_graph_after_gptq(fxp_model,
                                      tg,
                                      add_bias=gptq_config.train_bias)

    tg_gptq.user_info.gptq_info_dict['loss'] = loss_list
    return tg_gptq
