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
from typing import List, Callable

import tensorflow as tf
from keras import Model

from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.gptq.common.gptq_graph import get_kernel_attribute_name_for_gptq
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper



class SoftQuantizerRegularization:
    """
    A class to handle the computation of soft quantizer regularization for GPTQ training.
    """

    def __init__(self, beta_scheduler: Callable[[int], float]):
        """
        Initializes the regularization computation object with a LinearDecay object.

        Args:
            beta_scheduler: a callable that accepts current time step and returns a corresponding beta value.
        """
        # Initializing the temperature decay according to the number of expected gradient steps
        self.beta_scheduler = beta_scheduler
        self.count_iter = tf.Variable(0.)


    def __call__(self, model: Model, entropy_reg: float):
        """
        Returns the soft quantizer regularization value for SoftRounding.

        Args:
            model: A model to be quantized with SoftRounding.
            entropy_reg: Entropy value to scale the quantizer regularization.

        Returns: Regularization value.
        """
        soft_reg_aux: List[tf.Tensor] = []
        b = self.beta_scheduler(self.count_iter.value())
        for layer in model.layers:
            if isinstance(layer, KerasTrainableQuantizationWrapper):
                kernel_attribute = get_kernel_attribute_name_for_gptq(layer_type=type(layer.layer),
                                                                      fw_info=DEFAULT_KERAS_INFO)

                st = layer.weights_quantizers[kernel_attribute].get_soft_targets()
                soft_reg_aux.append(tf.reduce_sum(1 - tf.pow(tf.math.abs(st - .5) * 2, b)))

        reg = 0

        for sq in soft_reg_aux:
            reg += sq

        self.count_iter.assign_add(1.0)

        return entropy_reg * reg
