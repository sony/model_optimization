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

from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.core.common.constants import RANGE_MIN, RANGE_MAX
from model_compression_toolkit.qat.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit import qunatizers_infrastructure as qi
from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig


class STEUniformWeightQuantizer(qi.BaseKerasQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: a quantization config class with attributes for the quantization.

        """
        super().__init__(quantization_config,
                         qi.QuantizationTarget.Weights,
                         [qi.QuantizationMethod.UNIFORM])
        self.max_values = quantization_config.weights_quantization_params[RANGE_MAX]
        self.min_values = quantization_config.weights_quantization_params[RANGE_MIN]
        self.min_max_shape = np.asarray(self.max_values).shape
        self.max = np.reshape(self.max_values,
                              [-1]) if self.quantization_config.weights_per_channel_threshold else float(
            self.max_values)
        self.min = np.reshape(self.min_values,
                              [-1]) if self.quantization_config.weights_per_channel_threshold else float(
            self.min_values)

        if self.quantization_config.weights_per_channel_threshold and self.quantization_config.weights_channels_axis not in [
            -1,
            len(self.min_max_shape) - 1]:
            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            self.perm_vec = list(np.arange(len(self.min_max_shape)))
            self.perm_vec[self.quantization_config.weights_channels_axis] = len(self.min_max_shape) - 1
            self.perm_vec[len(self.min_max_shape) - 1] = self.quantization_config.weights_channels_axis
        else:
            self.perm_vec = None

        self.quantizer_parameters = {}

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        """
        Add min and max variables to layer.
        Args:
            tensor_shape: Tensor shape the quantizer quantize.
            name: Prefix of variables names.
            layer: Layer to add the variables to. The variables are saved
            in the layer's scope.

        Returns:
            Dictionary of new variables.
        """
        fq_min = layer.add_weight(
            name + FQ_MIN,
            shape=len(self.min) if self.quantization_config.weights_per_channel_threshold else (),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=False)
        fq_min.assign(self.min)

        fq_max = layer.add_weight(
            name + FQ_MAX,
            shape=len(self.max) if self.quantization_config.weights_per_channel_threshold else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        fq_max.assign(self.max)

        # save the quantizer added parameters for later calculations
        self.quantizer_parameters = {FQ_MIN: fq_min, FQ_MAX: fq_max}
        return self.quantizer_parameters

    def __call__(self, inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        _min = self.quantizer_parameters[FQ_MIN]
        _max = self.quantizer_parameters[FQ_MAX]
        if self.quantization_config.weights_per_channel_threshold:
            if self.perm_vec:
                inputs = tf.transpose(inputs, perm=self.perm_vec)
            q_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs, _min, _max,
                                                                                num_bits=self.quantization_config.weights_n_bits)
            if self.perm_vec:
                q_tensor = tf.transpose(q_tensor, perm=self.perm_vec)
        else:
            q_tensor = tf.quantization.fake_quant_with_min_max_vars(inputs, _min, _max,
                                                                    num_bits=self.quantization_config.weights_n_bits)

        return q_tensor
