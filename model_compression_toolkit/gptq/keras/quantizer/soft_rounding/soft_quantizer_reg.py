# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import List

import tensorflow as tf
from keras import Model

from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR, SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, \
    SOFT_ROUNDING_BETA, MAX_ITERATIONS_DEFAULT, GPTQ_ITER
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.quantizers_infrastructure import KerasQuantizationWrapper
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils


class LinearTempDecay:
    """
    Annealing process for the soft quantizer regularization temperature term.
    """

    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 20, end_b: int = 2):
        """
        Initializes a LinearTempDecay object.

        Args:
            t_max: maximal time step.
            rel_start_decay: Decay step size at the beginning of the process.
            start_b: Starting value of the regularization term.
            end_b: Target value of the regularization term.
        """

        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t: int) -> float:
        """
        Cosine annealing scheduler for soft quantizer regularization temperature term.

        Args:
            t: The current time step.

        Returns: Scheduled temperature.
        """

        is_before_start_decay = tf.cast(t < self.start_decay, tf.float32)

        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)

        return self.start_b * is_before_start_decay + \
               (1 - is_before_start_decay) * \
               (self.end_b + (self.start_b - self.end_b) * tf.math.maximum(0.0, (1 - rel_t)))


def soft_quantizer_regularization(model: Model, entropy_reg: float, n_batches: int, n_epochs: int) -> float:
    """
    Returns the soft quantizer regularization value for SoftRounding.

    Args:
        model: A model to be quantized with SoftRounding.
        entropy_reg: Entropy value to scale the quantizer regularization.

    Returns: Regularization value.
    """

    # gamma and zeta are stretch parameters for computing the rectified sigmoind function.
    # beta is used to set the regularization term.
    # See: https://arxiv.org/pdf/2004.10568.pdf
    gamma = SOFT_ROUNDING_GAMMA
    zeta = SOFT_ROUNDING_ZETA

    # Initializing the temperature decay according to the number of expected gradient steps
    init_decay = MAX_ITERATIONS_DEFAULT if n_batches is None else n_epochs * n_batches
    linear_decay = LinearTempDecay(init_decay)

    soft_reg_aux: List[tf.Tensor] = []
    for layer in model.layers:
        if isinstance(layer, KerasQuantizationWrapper):
            kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                  fw_info=DEFAULT_KERAS_INFO)

            soft_targets = tf.sigmoid(layer.weights_quantizers[kernel_attribute].quantizer_parameters[AUXVAR])
            st = qutils.clip(soft_targets * (zeta - gamma) + gamma, 1, 0)
            b = linear_decay(layer.weights_quantizers[kernel_attribute].quantizer_parameters[GPTQ_ITER].value())

            soft_reg_aux.append(tf.reduce_sum(1 - tf.pow(tf.math.abs(st - .5) * 2, b)))

    reg = 0

    for sq in soft_reg_aux:
        reg += sq

    return entropy_reg * reg
