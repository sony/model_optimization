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

from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.core.keras.quantizer.base_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as  qutils
from model_compression_toolkit.core.common.constants import THRESHOLD
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.gptq.keras.quantizer.kernel_functions import get_kernel
from model_compression_toolkit.gptq.common import gptq_constants


def symmetric_constrained_quantizer(input_tensor: tf.Tensor,
                                    auxvar_tensor: tf.Variable,
                                    max_tensor: tf.Tensor,
                                    num_bits: int,
                                    signed: bool,
                                    power_of_two: bool,
                                    max_lsbs_change: int = 1) -> tf.Tensor:
    """
    Quantize a tensor symmetrically with maximum LSBs shift.
    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift the weight due to gptq
        max_tensor: Tensor with max values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.
        max_lsbs_change: maximum number of LSBs that the auxvar is allowed to change

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        max_tensor = qutils.power_of_two_max(max_tensor)
    delta = qutils.calculate_delta(max_tensor, num_bits, signed)
    input_tensor_int = tf.stop_gradient(tf.round(input_tensor / delta))
    tensor_q = qutils.ste_round(
        input_tensor_int + qutils.ste_clip(auxvar_tensor, max_val=max_lsbs_change * delta) / delta)
    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1
    return delta * qutils.ste_clip(tensor_q, max_val=max_int, min_val=min_int)


class STEWeightQuantizer(BaseTrainableQuantizer):
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
            name + gptq_constants.GPTQ_ITER,
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)

        ptq_threshold_tensor = layer.add_weight(
            name + gptq_constants.THRESHOLD_TENSOR,
            shape=len(self.threshold_values) if self.per_axis else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values)

        auxvar_tensor = layer.add_weight(
            name + gptq_constants.AUXVAR,
            shape=w_shape,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        # save the quantizer added parameters for later calculations
        self.quantizer_parameters = {gptq_constants.THRESHOLD_TENSOR: ptq_threshold_tensor,
                                     gptq_constants.AUXVAR: auxvar_tensor,
                                     gptq_constants.GPTQ_ITER: ar_iter}
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

        auxvar = weights[gptq_constants.AUXVAR]
        ptq_threshold_tensor = weights[gptq_constants.THRESHOLD_TENSOR]

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

    def get_aux_variable(self) -> tf.Tensor:
        return self.quantizer_parameters[gptq_constants.AUXVAR]

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

    def get_quant_config(self, layer) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        old_threshold = self.quantizer_parameters[gptq_constants.THRESHOLD_TENSOR]
        return {THRESHOLD: old_threshold.numpy().reshape(self.threshold_shape)}

    def get_trainable_parameters(self):
        """
        A function to get a list trainable of trainable parameters of the quantizer for GPTQ retraining

        Returns:
            A list of trainable Tensors

        """
        return [t for t in self.quantizer_parameters.values() if t.trainable]

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
         This function return a list of quantizer parameters.
         Returns: A list of the quantizer parameters

         """
        return [self.quantizer_parameters[gptq_constants.THRESHOLD_TENSOR]]

    def __eq__(self, other: Any) -> bool:
        """
        Check if equals to another object.
        Args:
            other: Other object to compare.

        Returns:
            Whether they are equal or not.
        """
        if not isinstance(other, STEWeightQuantizer):
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
