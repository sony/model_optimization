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
import tensorflow as tf
import numpy as np

from model_compression_toolkit import GumbelConfig
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.base_gumbel_rounding import GumbelRoundingBase, \
    init_aux_var
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from typing import Dict, Any, List
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.gumbel_softmax import gumbel_softmax, ste_gumbel
from model_compression_toolkit.core.common.constants import RANGE_MIN, RANGE_MAX
from model_compression_toolkit.gptq.common import gptq_constants
from model_compression_toolkit.gptq.keras.quantizer.ste_rounding.uniform_ste import rounding_uniform_quantizer


def gumbel_rounding_uniform_quantizer(tensor_data: tf.Tensor,
                                      auxvar_tensor: tf.Variable,
                                      range_min: tf.Tensor,
                                      range_max: tf.Tensor,
                                      n_bits: int) -> tf.Tensor:
    """
    Quantize a tensor according to given range (min, max) and number of bits.

    Args:
        tensor_data: Tensor values to quantize.
        auxvar_tensor: Tensor that manifests the bit shift the weight due to gptq.
        range_min: minimum bound of the range for quantization (or array of min values per channel).
        range_max: maximum bound of the range for quantization (or array of max values per channel).
        n_bits: Number of bits to quantize the tensor.

    Returns:
        Quantized data.
    """

    # adjusts the quantization rage so the quantization grid include zero.
    a, b = qutils.fix_range_to_include_zero(range_min, range_max, n_bits)

    # Compute the step size of quantized values.
    delta = (b - a) / (2 ** n_bits - 1)

    input_tensor_int = tf.stop_gradient(tf.floor((tensor_data - a) / delta))  # Apply rounding
    tensor_q = input_tensor_int + auxvar_tensor

    # Clip data in range
    clipped_tensor = qutils.ste_clip(tensor_q, min_val=0, max_val=2 ** n_bits - 1)

    # Quantize the data between min/max of quantization range.
    q = delta * clipped_tensor + a
    return q


class UniformGumbelRounding(GumbelRoundingBase):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """
    PTQ_MIN_RANGE = "_min_range"
    PTQ_MAX_RANGE = "_max_range"

    def __init__(self, num_bits: int, per_axis: bool, signed: bool, quantization_parameter_learning: bool,
                 min_range: np.ndarray, max_range: np.ndarray, gumbel_config: GumbelConfig,
                 quantization_axis: int = -1, max_lsbs_change_map: dict = DefaultDict({}, lambda: 1),
                 max_iteration: int = 10000):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            num_bits: Number of bits to use for the quantization.
            per_axis: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            quantization_parameter_learning: Threshold to use for the quantization.
            min_range: a numpy array of the min range.
            max_range: a numpy array of the max range.
            gumbel_config: A class with the gumbel rounding configurations.
            quantization_axis: Axis of tensor to use for the quantization.
            max_lsbs_change_map: a mapping between number of bits to max lsb change.
            max_iteration: The number of iteration of gptq.
        """
        super().__init__(num_bits, per_axis, signed, False, False, quantization_parameter_learning,
                         quantization_axis, gumbel_config,
                         max_lsbs_change_map,
                         max_iteration)
        self.threshold_shape = np.asarray(min_range).shape
        self.min_range = np.reshape(np.asarray(min_range), [-1]) if self.per_axis else float(
            min_range)
        self.max_range = np.reshape(np.asarray(max_range), [-1]) if self.per_axis else float(
            max_range)
        self.k_threshold = len(self.max_range) if self.per_axis else 1

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
        super().build(tensor_shape, name, layer)

        if self.per_axis:
            input_shape = tensor_shape
            n_axis = len(input_shape)
            quantization_axis = n_axis + self.quantization_axis if self.quantization_axis < 0 else \
                self.quantization_axis
            reshape_shape = [self.k_threshold if i == quantization_axis else 1 for i in range(n_axis)]
        else:
            reshape_shape = [self.k_threshold]

        max_range = layer.add_weight(
            name + self.PTQ_MAX_RANGE,
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=self.quantization_parameter_learning)
        max_range.assign(self.max_range.reshape(reshape_shape))

        min_range = layer.add_weight(
            name + self.PTQ_MIN_RANGE,
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=self.quantization_parameter_learning)
        min_range.assign(self.min_range.reshape(reshape_shape))

        auxvar_tensor = layer.add_weight(
            name + gptq_constants.AUXVAR,
            shape=[self.m, *self.w_shape],
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)
        w = getattr(layer.layer, name)

        q_error = w - rounding_uniform_quantizer(w, min_range, max_range,
                                                 n_bits=self.num_bits)
        ceil_indicator = (q_error < 0).numpy().astype("int")  # Negative error means the choose point is rounded to ceil.
        auxvar_tensor.assign(init_aux_var(ceil_indicator, self.w_shape, self.m))

        self.quantizer_parameters.update({gptq_constants.AUXVAR: auxvar_tensor,
                                          self.PTQ_MAX_RANGE: max_range,
                                          self.PTQ_MIN_RANGE: min_range})
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
        ar_iter = weights[gptq_constants.GPTQ_ITER]
        ptq_min_range = weights[self.PTQ_MIN_RANGE]
        ptq_max_range = weights[self.PTQ_MAX_RANGE]
        aux_index_shift = weights[gptq_constants.AUXSHIFT]
        self.update_iteration(training, ar_iter)
        if self.per_axis:
            input_shape = inputs.shape
            n_axis = len(input_shape)
            quantization_axis = n_axis + self.quantization_axis if self.quantization_axis < 0 else \
                self.quantization_axis
            reshape_shape = [-1 if i == quantization_axis else 1 for i in range(n_axis)]

            reshape_shape_aux_ind = [-1, *[1 for _ in range(n_axis)]]
            #####################################################
            # Gumbel Softmax
            #####################################################
            if training:
                p_t = gumbel_softmax(auxvar, self.tau, self.g_t)
            else:
                p_t = gumbel_softmax(auxvar, self.minimal_temp, 0)
                p_t = ste_gumbel(p_t)
            self.p_t = p_t
            #####################################################
            # Calculate v hat and threshold hat
            #####################################################
            ptq_min_range = tf.reshape(ptq_min_range, reshape_shape)
            ptq_max_range = tf.reshape(ptq_max_range, reshape_shape)

            auxvar_hat = tf.reduce_sum(p_t * tf.reshape(aux_index_shift, reshape_shape_aux_ind), axis=0)
            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = gumbel_rounding_uniform_quantizer(inputs, auxvar_hat,
                                                         ptq_min_range,
                                                         ptq_max_range,
                                                         self.num_bits)
            return q_tensor
        else:
            raise NotImplemented
            return gumbel_rounding_uniform_quantizer(inputs, auxvar_hat,
                                                     ptq_max_range,
                                                     ptq_min_range,
                                                     self.num_bits)

    def get_quant_config(self, layer) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        min_range = self.quantizer_parameters[self.PTQ_MIN_RANGE]
        max_range = self.quantizer_parameters[self.PTQ_MAX_RANGE]
        return {RANGE_MIN: min_range.numpy().reshape(self.threshold_shape),
                RANGE_MAX: max_range.numpy().reshape(self.threshold_shape)}

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
         This function return a list of quantizer parameters.
         Returns: A list of the quantizer parameters

         """
        return [self.quantizer_parameters[self.PTQ_MIN_RANGE], self.quantizer_parameters[self.PTQ_MAX_RANGE]]
