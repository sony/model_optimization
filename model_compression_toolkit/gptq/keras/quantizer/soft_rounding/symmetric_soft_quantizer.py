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

import tensorflow as tf
import numpy as np

from model_compression_toolkit.core.common import max_power_of_two
from model_compression_toolkit.gptq.common.gptq_constants import PTQ_THRESHOLD, SCALE_PTQ, N_EPOCHS, \
    MAX_ITERATIONS_DEFAULT, SOFT_ROUNDING_GAMMA, SOFT_ROUNDING_ZETA, SOFT_ROUNDING_BETA, GPTQ_ITER, AUXVAR
from model_compression_toolkit.gptq.keras.quantizer import quant_utils as qutils
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape
from typing import Dict, Any, List
from model_compression_toolkit.core.common.constants import THRESHOLD, MIN_THRESHOLD
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.keras.quantizer.base_quantizer import BaseTrainableQuantizer



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


class SymmetricSoftRounding(BaseTrainableQuantizer):
    """
    Trainable symmetric quantizer to optimize the rounding of the quantized values using a soft quantization method.
    """

    def __init__(self, num_bits: int,
                 per_channel: bool,
                 signed: bool,
                 power_of_two: bool,
                 n_batches: int,
                 quantization_parameter_learning: bool,
                 threshold_values: np.ndarray,
                 quantization_axis: int = -1,
                 n_epochs: int = N_EPOCHS):
        """
        Initialize a SymmetricSoftRounding object with parameters to use
        for the quantization.
        Args:
            num_bits: Number of bits to use for the quantization.
            per_channel: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            power_of_two: Whether the threshold should be constrained or not.
            n_batches: The expected number of batches for each trainig epoch.
            quantization_parameter_learning: Whether to train the quantization threshold.
            threshold_values: Threshold to use for the quantization.
            quantization_axis: Axis of tensor to use for the quantization.
            n_epochs: Number of epochs to run training for.
        """

        super().__init__()
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.signed = signed
        self.power_of_two = power_of_two
        self.quantization_parameter_learning = quantization_parameter_learning
        self.quantization_axis = quantization_axis
        self.threshold_shape = np.asarray(threshold_values).shape
        self.threshold_values = np.reshape(np.asarray(threshold_values), [-1]) if self.per_channel else np.asarray(
            threshold_values)
        self.num_channels = len(self.threshold_values) if self.per_channel else 1

        # gamma and zeta are stretch parameters for computing the rectified sigmoind function.
        # beta is used to set the regularization term.
        # See: https://arxiv.org/pdf/2004.10568.pdf
        self.gamma = SOFT_ROUNDING_GAMMA
        self.zeta = SOFT_ROUNDING_ZETA
        self.beta = SOFT_ROUNDING_BETA

        self.quantizer_parameters = {}

        # Initializing the temperature decay according to the number of expected gradient steps
        if n_batches is None:
            Logger.warning(f"Number of batches is not set correctly for the Soft Quantizer. A default value of "  # pragma: no cover
                           f"{MAX_ITERATIONS_DEFAULT} is used to set the temperature decay which may affect the results.")

        init_decay = MAX_ITERATIONS_DEFAULT if n_batches is None else n_epochs * n_batches
        self.linear_decay = LinearTempDecay(init_decay)

    def build(self,
              tensor_shape: TensorShape,
              name: str,
              layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        """
        Add variables to the quantizer.

        Args:
            tensor_shape: Tensor shape the quantizer quantize.
            name: Prefix of variables names.
            layer: Layer to add the variables to. The variables are saved
            in the layer's scope.

        Returns:
            Dictionary of new variables.
        """

        super().build(tensor_shape, name, layer)

        if self.per_channel:
            reshape_shape = self._get_threshold_reshape_shape(tensor_shape, quant_axis_dim=self.num_channels)
        else:
            reshape_shape = [self.num_channels]

        ar_iter = layer.add_weight(
            f"{name}_{GPTQ_ITER}",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)

        ptq_threshold_tensor = layer.add_weight(
            f"{name}_{PTQ_THRESHOLD}",
            shape=reshape_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        ptq_threshold_tensor.assign(self.threshold_values.reshape(reshape_shape))

        w = getattr(layer.layer, name)
        auxvar_tensor = layer.add_weight(
            f"{name}_{AUXVAR}",
            shape=[*w.shape],
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)

        delta = qutils.calculate_delta(ptq_threshold_tensor, self.num_bits, self.signed)
        w_floor = tf.floor(w / delta)
        rest = (w / delta) - w_floor  # rest of rounding [0, 1)
        # Note that (rest - self.gamma) can't be zero since rest is positive and gamma is negative, so the division
        # is safe
        alpha = -qutils.safe_log((self.zeta - self.gamma) / (rest - self.gamma) - 1, 1e-16)  # => sigmoid(alpha) = rest

        auxvar_tensor.assign(alpha)

        self.quantizer_parameters.update({AUXVAR: auxvar_tensor,
                                          PTQ_THRESHOLD: ptq_threshold_tensor,
                                          GPTQ_ITER: ar_iter})

        if self.quantization_parameter_learning:
            scale = layer.add_weight(
                f"{name}_{SCALE_PTQ}",
                shape=self.num_channels,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True)
            self.quantizer_parameters.update({SCALE_PTQ: scale})

        return self.quantizer_parameters

    def get_quantization_variable(self) -> List[tf.Tensor]:
        """
        Returns:
            A list of the quantization parameters (if there are defined parameters for the quantizer).
        """

        if self.quantization_parameter_learning and not self.power_of_two:
            return [self.quantizer_parameters[SCALE_PTQ]]
        else:
            return []

    def get_regularization(self) -> tf.Tensor:
        """
        Computes the regularization term for the soft rounding loss.

        Returns:
            regularization term.
        """

        st = self.get_soft_targets()
        b = self.linear_decay(self.ar_iter.value())
        return tf.reduce_sum(1 - tf.pow(tf.math.abs(st - .5) * 2, b))

    def get_trainable_parameters(self) -> List[tf.Tensor]:
        """
        A function to get a list trainable of trainable parameters of the quantizer for GPTQ retraining

        Returns:
            A list of trainable Tensors
        """
        return [t for t in self.quantizer_parameters.values() if t.trainable]

    def get_config(self) -> Dict[str, Any]:
        """
        Returns:
            Configuration of SymmetricSoftRounding.
        """

        return {
            'num_bits': self.num_bits,
            'per_channel': self.per_channel,
        }

    def get_soft_targets(self) -> tf.Tensor:
        """
        Computes the rectified sigmoid function for the quantization target parameters.

        Returns:
            A tensor with the soft rounding targets values.

        """
        return qutils.clip(
            tf.sigmoid(self.quantizer_parameters[AUXVAR]) * (self.zeta - self.gamma) + self.gamma, 1, 0)

    def get_aux_variable(self) -> tf.Tensor:
        """
        Returns:
            The auxiliary variable of the rounding learning.
        """
        return self.quantizer_parameters[AUXVAR]

    def __call__(self, inputs: tf.Tensor,
                 training: bool,
                 weights: Dict[str, tf.Variable],
                 **kwargs: Dict[str, Any]) -> tf.Tensor:
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

        self.ar_iter = weights[GPTQ_ITER]
        ptq_threshold_tensor = weights[PTQ_THRESHOLD]

        if self.per_channel:
            reshape_shape = self._get_threshold_reshape_shape(inputs.shape, quant_axis_dim=-1)

            ##########################################################
            # Calculate soft rounding targets and optimized threshold
            ##########################################################
            ptq_threshold_tensor_hat = tf.reshape(ptq_threshold_tensor, reshape_shape)
            aux_var = self.get_soft_targets()

            #####################################################
            # Soft Rounding
            #####################################################
            if training:
                self.ar_iter.assign_add(1.0)
            else:
                aux_var = tf.cast(weights[AUXVAR] >= 0, tf.float32)

            #####################################################
            # Quantized Input
            #####################################################
            q_tensor = qutils.symmetric_rounding_quantizer(input_tensor=inputs,
                                                           auxvar_tensor=aux_var,
                                                           threshold_tensor=ptq_threshold_tensor_hat,
                                                           num_bits=self.num_bits,
                                                           signed=self.signed,
                                                           power_of_two=self.power_of_two)

            if self.quantization_parameter_learning and not self.power_of_two:
                scale = tf.reshape(self.quantizer_parameters[SCALE_PTQ], reshape_shape)
                q_tensor *= scale

            return q_tensor
        else:
            return qutils.symmetric_rounding_quantizer(input_tensor=inputs,
                                                       auxvar_tensor=weights[AUXVAR],
                                                       threshold_tensor=ptq_threshold_tensor.value(),
                                                       num_bits=self.num_bits,
                                                       signed=self.signed,
                                                       power_of_two=self.power_of_two)

    # TODO: Extract this method to a parent class of all GPTQ quantizer and use it in other quantizers (such as STE)
    def _get_threshold_reshape_shape(self, tensor_shape, quant_axis_dim):
        """
        Gets a shape that contains 1 in all axis except the quantization axis, to adjust the threshold tensor for
        per-channel quantization.

        Args:
            tensor_shape: The shape of the tensor to be quantize.
            quant_axis_dim: The dimension of the quantization axis.

        Returns: A shape to reshape the threshold tensor according to.

        """
        n_axis = len(tensor_shape)
        quantization_axis = n_axis + self.quantization_axis if self.quantization_axis < 0 else \
            self.quantization_axis

        return [quant_axis_dim if i == quantization_axis else 1 for i in range(n_axis)]

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
            old_threshold = self.quantizer_parameters[PTQ_THRESHOLD]
            old_threshold = max_power_of_two(old_threshold, MIN_THRESHOLD)

        else:
            old_threshold = self.quantizer_parameters[PTQ_THRESHOLD]
            if self.quantization_parameter_learning:
                scale = tf.reshape(self.quantizer_parameters[SCALE_PTQ], self.threshold_shape)
                old_threshold = old_threshold * scale
            old_threshold = old_threshold.numpy()
        old_threshold = old_threshold.reshape(self.threshold_shape)
        return {THRESHOLD: old_threshold}
