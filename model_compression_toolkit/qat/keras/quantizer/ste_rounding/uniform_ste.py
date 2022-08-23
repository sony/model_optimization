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
from model_compression_toolkit.core.keras.quantizer.base_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MIN, RANGE_MAX
from model_compression_toolkit.qat.common import THRESHOLD_TENSOR
from model_compression_toolkit.qat.common.constants import FQ_MIN, FQ_MAX


class STEUniformWeightQuantizer(BaseTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self,
                 num_bits: int,
                 per_axis: bool,
                 signed: bool,
                 min_values: np.ndarray,
                 max_values: np.ndarray,
                 quantization_axis: int = -1):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            num_bits: Number of bits to use for the quantization.
            per_axis: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            min_values: Minimum values to use for the quantization.
            max_values: Maximum to use for the quantization.
            quantization_axis: Axis of tensor to use for the quantization.
        """
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.signed = signed
        self.min_max_shape = np.asarray(max_values).shape
        self.max_values = max_values
        self.min_values = min_values
        self.max = np.reshape(max_values, [-1]) if self.per_axis else float(max_values)
        self.min = np.reshape(min_values, [-1]) if self.per_axis else float(min_values)
        self.quantization_axis = quantization_axis

        if self.per_axis and self.quantization_axis not in [-1, len(self.min_max_shape)-1]:
            # Tensorflow's fake_quant_with_min_max_vars_per_channel only works on last axis, so
            # need to move the quantization axis to the last axis
            self.perm_vec = list(np.arange(len(self.min_max_shape)))
            self.perm_vec[self.quantization_axis] = len(self.min_max_shape)-1
            self.perm_vec[len(self.min_max_shape)-1] = self.quantization_axis
        else:
            self.perm_vec = None

        self.quantizer_parameters = {}

    def build(self,
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
            shape=len(self.min) if self.per_axis else (),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=False)
        fq_min.assign(self.min)

        fq_max = layer.add_weight(
            name + FQ_MAX,
            shape=len(self.max) if self.per_axis else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        fq_max.assign(self.max)

        # save the quantizer added parameters for later calculations
        self.quantizer_parameters = {FQ_MIN: fq_min, FQ_MAX: fq_max}
        return self.quantizer_parameters

    def __call__(self, inputs: tf.Tensor,
                 training: bool,
                 weights: Dict[str, tf.Variable],
                 **kwargs: Dict[str, Any]):
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.
            weights: Dictionary of weights the quantizer can use to quantize the tensor.
            **kwargs: Additional variables the quantizer may receive.

        Returns:
            The quantized tensor.
        """

        _min = weights[FQ_MIN]
        _max = weights[FQ_MAX]
        if self.per_axis:
            if self.perm_vec:
                inputs = tf.transpose(inputs, perm=self.perm_vec)
            q_tensor = tf.quantization.fake_quant_with_min_max_vars_per_channel(inputs, _min, _max,
                                                                                num_bits=self.num_bits)
            if self.perm_vec:
                q_tensor = tf.transpose(q_tensor, perm=self.perm_vec)
        else:
            q_tensor = tf.quantization.fake_quant_with_min_max_vars(inputs, _min, _max,
                                                                    num_bits=self.num_bits)

        return q_tensor

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: Configuration of TrainableQuantizer.
        """

        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'signed': self.signed,
            'min_values': self.min_values,
            'max_values': self.max_values,
            'quantization_axis': self.quantization_axis
        }

    def get_quant_config(self, layer) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after QAT retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during QAT retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        return {RANGE_MIN: self.quantizer_parameters[FQ_MIN].numpy().reshape(self.min_max_shape),
                RANGE_MAX: self.quantizer_parameters[FQ_MAX].numpy().reshape(self.min_max_shape)}

    def get_trainable_parameters(self) -> List[tf.Tensor]:
        """
        A function to get a list trainable of trainable parameters of the quantizer for QAT retraining

        Returns:
            A list of trainable Tensors

        """
        return [t for t in self.quantizer_parameters.values() if t.trainable]

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
         This function return a list of quantizer parameters.
         Returns: A list of the quantizer parameters

         """
        return [self.quantizer_parameters[FQ_MIN], self.quantizer_parameters[FQ_MAX]]

    def __eq__(self, other: Any) -> bool:
        """
        Check if equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are equal or not.
        """
        if not isinstance(other, STEUniformWeightQuantizer):
            return False

        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.signed == other.signed)

    def __ne__(self, other: Any) -> bool:
        """
        Check if not equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are different or not.
        """
        return not self.__eq__(other)
