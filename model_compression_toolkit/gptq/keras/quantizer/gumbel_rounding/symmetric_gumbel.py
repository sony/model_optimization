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
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.base_gumbel_rounding import GumbelRoundingBase
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from typing import Dict, Any, List
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.gumbel_softmax import gumbel_softmax, ste_gumbel
from model_compression_toolkit.core.common.constants import THRESHOLD, GUMBEL_MAX_ITER, MIN_THRESHOLD
from model_compression_toolkit.gptq.common import gptq_constants
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import max_power_of_two

def gumbel_rounding_symmetric_quantizer(input_tensor: tf.Tensor,
                                        auxvar_tensor: tf.Variable,
                                        max_tensor: tf.Tensor,
                                        num_bits: int,
                                        signed: bool,
                                        power_of_two: bool) -> tf.Tensor:
    """
    Quantize a tensor symmetrically with maximum LSBs shift.
    Args:
        input_tensor: Tensor to quantize. values of this tensor are not changed during gptq.
        auxvar_tensor: Tensor that manifests the bit shift the weight due to gptq.
        max_tensor: Tensor with max values to compute the threshold.
        num_bits: Num of bits to use.
        signed: Signedness of the quantization range.
        power_of_two: Whether the threshold should be constrained or not.

    Returns:
        A quantized tensor.
    """

    if power_of_two:
        max_tensor = qutils.power_of_two_max(max_tensor)
    delta = qutils.calculate_delta(max_tensor, num_bits, signed)
    input_tensor = tf.stop_gradient(input_tensor)
    input_tensor_int = qutils.ste_round(input_tensor / delta)
    tensor_q = input_tensor_int + auxvar_tensor
    min_int = -int(signed) * (2 ** (num_bits - int(signed)))
    max_int = (2 ** (num_bits - int(signed))) - 1
    return delta * qutils.clip(tensor_q, max_val=max_int, min_val=min_int)

class SymmetricGumbelRounding(GumbelRoundingBase):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """
    PTQ_THRESHOLD = "_ptq_threshold"
    SCALE_PTQ = "_scale"

    def __init__(self, num_bits: int,
                 per_axis: bool,
                 signed: bool,
                 power_of_two: bool,
                 quantization_parameter_learning: bool,
                 threshold_values: np.ndarray,
                 gumbel_config: GumbelConfig,
                 quantization_axis: int = -1,
                 max_lsbs_change_map: dict = DefaultDict({}, lambda: 1),
                 max_iteration: int = GUMBEL_MAX_ITER,
                 gumbel_scale: float = 1.0):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            num_bits: Number of bits to use for the quantization.
            per_axis: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            threshold_values: Threshold to use for the quantization.
            gumbel_config: A class with the gumbel rounding configurations.
            quantization_axis: Axis of tensor to use for the quantization.
            power_of_two: Whether the threshold should be constrained or not.
            max_lsbs_change_map: a mapping between number of bits to max lsb change.
            max_iteration: The number of iteration of gptq.
            gumbel_scale: A normalization factor for the gumbel tensor values
        """
        super().__init__(num_bits, per_axis, signed, True, power_of_two, quantization_parameter_learning,
                         quantization_axis, gumbel_config,
                         max_lsbs_change_map,
                         max_iteration)
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_axis else float(
            threshold_values)
        self.k_threshold = len(self.threshold_values) if self.per_axis else 1
        self.gumbel_scale = gumbel_scale

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

        ptq_threshold_tensor = layer.add_weight(
            name + self.PTQ_THRESHOLD,
            shape=self.k_threshold,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values)

        self.quantizer_parameters.update({self.PTQ_THRESHOLD: ptq_threshold_tensor})

        if self.quantization_parameter_learning and not self.power_of_two:
            scale = layer.add_weight(
                name + self.SCALE_PTQ,
                shape=self.k_threshold,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True)
            self.quantizer_parameters.update({self.SCALE_PTQ: scale})

        return self.quantizer_parameters

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
        This function return a list of quantizer parameters.
        Returns: A list of the quantizer parameters

        """
        if self.quantization_parameter_learning and not self.power_of_two:
            return [self.quantizer_parameters[self.SCALE_PTQ]]
        else:
            return [self.quantizer_parameters[self.PTQ_THRESHOLD]]

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
        ptq_threshold_tensor = weights[self.PTQ_THRESHOLD]
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
                p_t = gumbel_softmax(auxvar, self.tau, self.g_t, gumbel_scale=self.gumbel_scale)
            else:
                p_t = gumbel_softmax(auxvar, self.minimal_temp, 0)
                p_t = ste_gumbel(p_t)
            self.p_t = p_t
            #####################################################
            # Calculate v hat and threshold hat
            #####################################################
            ptq_threshold_tensor_hat = tf.reshape(ptq_threshold_tensor, reshape_shape)
            auxvar_hat = tf.reduce_sum(p_t * tf.reshape(aux_index_shift, reshape_shape_aux_ind), axis=0)
            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = gumbel_rounding_symmetric_quantizer(inputs, auxvar_hat,
                                                           ptq_threshold_tensor_hat,
                                                           self.num_bits,
                                                           self.signed,
                                                           self.power_of_two)
            if self.quantization_parameter_learning and not self.power_of_two:
                scale = tf.reshape(self.quantizer_parameters[self.SCALE_PTQ], reshape_shape)
                q_tensor *= scale

            return q_tensor
        else:
            return gumbel_rounding_symmetric_quantizer(inputs, auxvar,
                                                       ptq_threshold_tensor,
                                                       self.num_bits,
                                                       self.signed,
                                                       self.power_of_two)

    def get_quant_config(self, layer) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Args:
            layer: quantized layer

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """

        if self.power_of_two:
            old_threshold = self.quantizer_parameters[self.PTQ_THRESHOLD]
            old_threshold = max_power_of_two(old_threshold, MIN_THRESHOLD)
        else:
            old_threshold = self.quantizer_parameters[self.PTQ_THRESHOLD]
            if self.quantization_parameter_learning:
                old_threshold = old_threshold * self.quantizer_parameters[self.SCALE_PTQ]
            old_threshold = old_threshold.numpy()
        old_threshold = old_threshold.reshape(self.threshold_shape)
        return {THRESHOLD: old_threshold}
