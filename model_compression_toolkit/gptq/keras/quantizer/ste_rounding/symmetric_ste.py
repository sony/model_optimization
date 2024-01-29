# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.gptq import RoundingType
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from mct_quantizers import QuantizationTarget
from model_compression_toolkit.gptq.common.gptq_constants import AUXVAR, PTQ_THRESHOLD
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from model_compression_toolkit.constants import THRESHOLD
from model_compression_toolkit.defaultdict import DefaultDict
from model_compression_toolkit.gptq.keras.quantizer.base_keras_gptq_quantizer import BaseKerasGPTQTrainableQuantizer
from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerWeightsConfig
from mct_quantizers import mark_quantizer
from model_compression_toolkit.trainable_infrastructure.common.quant_utils import \
    get_threshold_reshape_shape
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup


def pertubation_symmetric_quantizer(input_tensor: tf.Tensor,
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


@mark_quantizer(quantization_target=QuantizationTarget.Weights,
                quantization_method=[QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC],
                identifier=RoundingType.STE)
class STEWeightGPTQQuantizer(BaseKerasGPTQTrainableQuantizer):
    """
    Trainable symmetric quantizer to quantize a layer weights.
    """

    def __init__(self,
                 quantization_config: TrainableQuantizerWeightsConfig,
                 max_lsbs_change_map: dict = DefaultDict(default_value=1)):
        """
        Initialize a STEWeightGPTQQuantizer object with parameters to use for the quantization.

        Args:
            quantization_config: Trainable weights quantizer config.
            max_lsbs_change_map: a mapping between number of bits to max lsb change.
        """
        super().__init__(quantization_config)
        self.num_bits = quantization_config.weights_n_bits
        self.per_channel = quantization_config.weights_per_channel_threshold

        threshold_values = quantization_config.weights_quantization_params[THRESHOLD]
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_channel else float(
            threshold_values)

        self.quantization_axis = quantization_config.weights_channels_axis
        self.power_of_two = quantization_config.weights_quantization_method == QuantizationMethod.POWER_OF_TWO
        self.max_lsbs_change = max_lsbs_change_map.get(self.num_bits)

    def initialize_quantization(self,
                                tensor_shape: Any,
                                name: str,
                                layer: Any):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """

        ptq_threshold_tensor = layer.add_weight(
            f"{name}_{PTQ_THRESHOLD}",
            shape=len(self.threshold_values) if self.per_channel else (),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values)

        w = getattr(layer.layer, name)
        auxvar_tensor = layer.add_weight(
            f"{name}_{AUXVAR}",
            shape=list(w.shape),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(PTQ_THRESHOLD, ptq_threshold_tensor, VariableGroup.QPARAMS)
        self.add_quantizer_variable(AUXVAR, auxvar_tensor, VariableGroup.WEIGHTS)

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.

        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        auxvar = self.get_quantizer_variable(AUXVAR)
        ptq_threshold_tensor = self.get_quantizer_variable(PTQ_THRESHOLD)

        if self.per_channel:
            reshape_shape = get_threshold_reshape_shape(inputs.shape,
                                                        quant_axis=self.quantization_axis,
                                                        quant_axis_dim=-1)
            ptq_threshold_tensor = tf.reshape(ptq_threshold_tensor, reshape_shape)
            q_tensor = pertubation_symmetric_quantizer(inputs,
                                                       auxvar,
                                                       ptq_threshold_tensor,
                                                       self.num_bits,
                                                       signed=True,
                                                       power_of_two=self.power_of_two,
                                                       max_lsbs_change=self.max_lsbs_change)
            return q_tensor
        else:
            return pertubation_symmetric_quantizer(inputs,
                                                   auxvar,
                                                   ptq_threshold_tensor,
                                                   self.num_bits,
                                                   signed=True,
                                                   power_of_two=self.power_of_two)


    def get_quant_config(self) -> Dict[str, np.ndarray]:
        """
        Returns the config used to edit NodeQuantizationConfig after GPTQ retraining

        Returns:
            A dictionary of attributes the quantize_config retraining has changed during GPTQ retraining.
            Keys must match NodeQuantizationConfig attributes

        """
        old_threshold = self.get_quantizer_variable(PTQ_THRESHOLD)
        return {THRESHOLD: old_threshold.numpy().reshape(self.threshold_shape)}
