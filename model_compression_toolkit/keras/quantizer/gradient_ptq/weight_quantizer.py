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

from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.keras.quantizer.base_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.keras.quantizer.gradient_ptq.utils import symmetric_constrained_quantizer
from model_compression_toolkit.common.constants import THRESHOLD
from model_compression_toolkit.common.defaultdict import DefaultDict
from model_compression_toolkit.keras.constants import KERNEL


def get_kernel(weights_list: list) -> tf.Tensor:
    """
    This function a list of weights and return the kernel
    Args:
        weights_list:  A list of Tensors

    Returns: The kernel tensor.

    """
    for w in weights_list:
        if KERNEL in w.name:
            return w
    raise Exception("Can't find kernel variable")


class TrainableWeightQuantizer(BaseTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self,
                 num_bits: int,
                 per_axis: bool,
                 signed: bool,
                 threshold_values: np.ndarray,
                 quantization_axis: int = -1,
                 power_of_two: bool = True,
                 max_lsbs_change_map: dict = DefaultDict({}, lambda: 1)):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            num_bits: Number of bits to use for the quantization.
            per_axis: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            threshold_values: Threshold to use for the quantization.
            quantization_axis: Axis of tensor to use for the quantization.
            power_of_two: Whether the threshold should be constrained or not.
            max_lsbs_change_map: a mapping between number of bits to max lsb change.
        """
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.signed = signed
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_axis else float(
            threshold_values)
        self.quantization_axis = quantization_axis
        self.power_of_two = power_of_two
        self.max_lsbs_change = max_lsbs_change_map.get(num_bits)
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
        w_shape = get_kernel(layer.weights).shape
        ar_iter = layer.add_weight(
            name + '_gptq_iter',
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)

        ptq_threshold_tensor = layer.add_weight(
            name + '_ptq_threshold',
            shape=len(self.threshold_values) if self.per_axis else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values)

        auxvar_tensor = layer.add_weight(
            name + '_auxvar',
            shape=w_shape,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        # save the quantizer added parameters for later calculations
        self.quantizer_parameters = {'ptq_threshold_tensor': ptq_threshold_tensor, 'auxvar_tensor': auxvar_tensor,
                                     'ar_iter': ar_iter}
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

        auxvar = weights['auxvar_tensor']
        ptq_threshold_tensor = weights['ptq_threshold_tensor']

        if self.per_axis:
            input_shape = inputs.shape
            n_axis = len(input_shape)
            quantization_axis = n_axis + self.quantization_axis if self.quantization_axis < 0 else \
                self.quantization_axis
            reshape_shape = [-1 if i == quantization_axis else 1 for i in range(n_axis)]
            ptq_threshold_tensor = tf.reshape(ptq_threshold_tensor, reshape_shape)
            q_tensor = symmetric_constrained_quantizer(inputs, auxvar,
                                                       ptq_threshold_tensor,
                                                       self.num_bits,
                                                       self.signed,
                                                       self.power_of_two,
                                                       max_lsbs_change=self.max_lsbs_change)
            return q_tensor
        else:
            return symmetric_constrained_quantizer(inputs, auxvar,
                                                   ptq_threshold_tensor,
                                                   self.num_bits,
                                                   self.signed,
                                                   self.power_of_two)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns: Configuration of TrainableQuantizer.
        """

        return {
            'num_bits': self.num_bits,
            'per_axis': self.per_axis,
            'symmetric': self.symmetric,
            'power_of_two': self.power_of_two
        }

    def calc_quant_config(self, layer):
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        old_threshold = self.quantizer_parameters['ptq_threshold_tensor']
        return {THRESHOLD: old_threshold.numpy().reshape(self.threshold_shape)}

    def get_trainable_parameters(self):
        """
        A function to get a list trainable of trainable parameters of the quantizer for GPTQ retraining

        Returns:
            A list of trainable Tensors

        """
        return [t for t in self.quantizer_parameters.values() if t.trainable]

    def __eq__(self, other: Any) -> bool:
        """
        Check if equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are equal or not.
        """
        if not isinstance(other, TrainableWeightQuantizer):
            return False

        return (self.num_bits == other.num_bits and
                self.per_axis == other.per_axis and
                self.symmetric == other.symmetric)

    def __ne__(self, other: Any) -> bool:
        """
        Check if not equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are differ or not.
        """
        return not self.__eq__(other)
