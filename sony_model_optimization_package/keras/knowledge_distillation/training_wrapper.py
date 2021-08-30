# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================
from typing import Callable, List, Tuple

import tensorflow as tf
from tqdm import tqdm

from sony_model_optimization_package import common
from sony_model_optimization_package.common.quantization.quantization_config import QuantizationConfig
from sony_model_optimization_package.common import Graph
from sony_model_optimization_package.keras.back2framework.model_builder import model_builder, ModelBuilderMode
from sony_model_optimization_package.keras.knowledge_distillation.graph_info import get_compare_points, \
    get_trainable_parameters
from sony_model_optimization_package.common.framework_info import FrameworkInfo
from sony_model_optimization_package.keras.knowledge_distillation.graph_update import update_graph_after_kd
from sony_model_optimization_package.keras.knowledge_distillation.knowledge_distillation_loss import \
    multiple_tensors_cs_loss, mve_loss
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
import numpy as np


class KnowledgeDistillationConfig(object):
    """
    Configuration to use for quantization with KnowledgeDistillation (experimental).
    """

    def __init__(self,
                 n_iter: int,
                 optimizer: OptimizerV2 = tf.keras.optimizers.Adam(learning_rate=0.0001),
                 log_function: Callable = None,
                 train_bias: bool = True,
                 representative_data_gen: Callable = None):
        """
        Initialize a KnowledgeDistillationConfig.

        Args:
            n_iter (int): Number of iterations to train.
            optimizer (OptimizerV2): Optimizer to use.
            log_function (Callable): Function to log information about the KD process.
            train_bias (bool): Whether to update the bias during the training or not.
            representative_data_gen (Callable): Dataset generator.

        Examples:
            Create a KnowledgeDistillationConfig to run for 5 iteration and uses a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]
            >>> kd_conf = KnowledgeDistillationConfig(n_iter=5, representative_data_gen=repr_datagen)

            An optimizer can be passed:

            >>> kd_conf = KnowledgeDistillationConfig(n_iter=5, representative_data_gen=repr_datagen, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.2))

            To disable the biases training, one may set train_bias to False (enabled by default):

            >>> kd_conf = KnowledgeDistillationConfig(n_iter=5, representative_data_gen=repr_datagen, train_bias=False)

            The configuration can then be passed to :func:`~sony_model_optimization_package.keras_post_training_quantization`.

        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.log_function = log_function
        self.train_bias = train_bias
        self.representative_data_gen = representative_data_gen


def knowledge_distillation_training_wrapper(tg: Graph,
                                            representative_data_gen: Callable,
                                            kd_config: KnowledgeDistillationConfig,
                                            fw_info: FrameworkInfo):
    """
    Build two models from a graph: A teacher network (float model) and a student network (quantized model).
    Use the dataset generator to pass images through the teacher and student networks to get intermediate
    layers outputs. Use the outputs to compute the observed loss and to backpropagate the error
    in the student network, to minimize it in the next similar steps.
    All parameters (such as number of iterations, optimizer, etc.) are in the passed KnowledgeDistillationConfig.

    Args:
        tg: Graph to build networks from.
        representative_data_gen: Dataset generator to get images.
        kd_config: KnowledgeDistillationConfig with parameters about the distillation process.
        fw_info: Framework information needed for keras kernel ops list.

    Returns:
        Graph of quantized model after KD training.
    """

    input_scale = 1
    kd_representative_data_gen = kd_config.representative_data_gen if kd_config.representative_data_gen is not None \
        else representative_data_gen
    #########################################
    # Build two models and compare points
    #########################################
    compare_points, _ = get_compare_points(tg)  # get compare points
    n = len(compare_points)
    float_model, float_user_info = model_builder(tg,
                                                 mode=ModelBuilderMode.FLOAT,
                                                 append2output=compare_points)
    fxp_model, kd_user_info = model_builder(tg,
                                            mode=ModelBuilderMode.KNOWLEDGEDISTILLATION,
                                            append2output=compare_points)

    trainable_weights = get_trainable_parameters(fxp_model,
                                                 fw_info,
                                                 add_bias=kd_config.train_bias)

    if float_user_info.input_scale != kd_user_info.input_scale:
        common.Logger.error("Input scale mismatch between float and kd networks")  # pragma: no cover
    else:
        input_scale = kd_user_info.input_scale
    #########################################
    # Optimization Loop
    #########################################
    alpha = tf.Variable(tf.ones(n))

    def update_step(input_data: List[np.ndarray]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Get outputs from both teacher and student networks. Compute the observed error,
        and use it to compute the gradients and applying them to the student weights.
        Args:
            input_data: Input to pass through the networks.

        Returns:
            Loss and gradients.
        """
        y_float = float_model(input_data)  # running float model
        with tf.GradientTape(persistent=True) as tape:
            y_fxp = fxp_model(input_data)  # running fxp model
            loss_alpha = 100 * mve_loss(y_fxp, y_float, alpha)
            loss_value = multiple_tensors_cs_loss(y_fxp, y_float, weights=tf.nn.softmax(alpha))  # calculate cs loss

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads_alpha = tape.gradient(loss_alpha, [alpha])
        grads = tape.gradient(loss_value, trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        kd_config.optimizer.apply_gradients(zip(grads, trainable_weights))
        kd_config.optimizer.apply_gradients(zip(grads_alpha, [alpha]))
        return loss_value, grads

    loss_list = []
    for _ in tqdm(range(kd_config.n_iter)):
        data = kd_representative_data_gen()
        loss_value_step, grads = update_step([d * input_scale for d in data])
        if kd_config.log_function is not None:
            kd_config.log_function(loss_value_step, grads, trainable_weights, alpha, compare_points)
        loss_list.append(loss_value_step.numpy())
        common.Logger.debug(f'last loss value:{loss_list[-1]}')

    #########################################
    # Update Graph after kd
    #########################################
    tg_kd = update_graph_after_kd(fxp_model,
                                  tg,
                                  add_bias=kd_config.train_bias)
    tg_kd.user_info.kd_info_dict['loss'] = loss_list
    for n in tg_kd.nodes():
        if n.weights_quantization_cfg is not None:
            n.weights_quantization_cfg.weights_bias_correction = n.weights_quantization_cfg.weights_bias_correction and not kd_config.train_bias  # Update bias correction flag
    return tg_kd
