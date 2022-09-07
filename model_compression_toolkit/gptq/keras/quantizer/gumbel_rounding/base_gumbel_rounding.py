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

from model_compression_toolkit import GumbelConfig
from model_compression_toolkit.core.keras.quantizer.base_quantizer import BaseTrainableQuantizer
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core import common
from model_compression_toolkit.gptq.keras.quantizer import kernel_functions
from model_compression_toolkit.gptq.keras.quantizer.gumbel_rounding.gumbel_softmax import sample_gumbel
from model_compression_toolkit.gptq.common import gptq_constants
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow.python.framework.tensor_shape import TensorShape

P_INIT = 0.01


def _init_aux_var(w_shape: List[int], m: int, p: float = P_INIT) -> np.ndarray:
    """
    This function generate a random pi matrix for Gumbel Rounding
    Args:
        w_shape(List[int]): A list of integers that represent the shape of the weights tensor to be quantization
        p(float): A floating point number that represent the probability of non round options of pi matrix.
        m(int):  An integer that define the number of shift

    Returns: A numpy array of pi tensor

    """
    m_hat = m // 2
    shift = -np.log(-np.log(1 - p))
    n = np.random.randn(*[m, *w_shape]) * np.sqrt(np.power(np.pi, 2) / 6)
    n[m_hat, :] += shift
    return n


def _init_shift_var(m: int) -> List[int]:
    """
    This function generate an list of 2*m+1 from -m to m
    Args:
        m: An integer value the represent m

    Returns: A list of size m

    """
    m_hat = m // 2
    aux_index_shift = [-m_hat + i for i in range(m)]
    return aux_index_shift


class GumbelRoundingBase(BaseTrainableQuantizer):
    def __init__(self,
                 num_bits: int,
                 per_axis: bool,
                 signed: bool,
                 symmetric: bool,
                 power_of_two: bool,
                 quantization_parameter_learning: bool,
                 quantization_axis: int,
                 gumbel_config: GumbelConfig,
                 max_lsbs_change_map: dict = DefaultDict({}, lambda: 1),
                 max_iteration: int = 10000):
        """
        A base class for GumRounding

        Args:
            num_bits: Number of bits to use for the quantization.
            per_axis: Whether to quantize per-channel or per-tensor.
            signed: Signedness to use for the quantization range.
            symmetric:  Whether to quantize is symmetric.
            power_of_two: Whether to quantize is power-of-two.
            quantization_parameter_learning: A bool flag state if the quantizer parameter are trainable
            quantization_axis: Axis of tensor to use for the quantization.
            gumbel_config: A class with the gumbel rounding configurations.
            max_lsbs_change_map: a mapping between number of bits to max lsb change.
            max_iteration: The number of iteration of gptq.
        """
        self.num_bits = num_bits
        self.per_axis = per_axis
        self.signed = signed
        self.quantization_axis = quantization_axis
        self.max_iteration = max_iteration
        self.power_of_two = power_of_two
        self.symmetric = symmetric
        self.quantization_parameter_learning = quantization_parameter_learning
        self.temperature_learning = gumbel_config.temperature_learning
        self.quantizer_parameters = {}
        self.gumbel_config = gumbel_config

        self.max_lsbs_change_map = max_lsbs_change_map
        self.max_lsbs_change = max_lsbs_change_map.get(num_bits)
        self.m = 2 * self.max_lsbs_change + 1

        self.n_cycles = gumbel_config.n_cycles
        self.minimal_temp = gumbel_config.minimal_temp
        self.maximal_temp = gumbel_config.maximal_temp
        self.cycle_iterations = int(self.max_iteration / self.n_cycles)
        self.tau = None
        self.g_t = None
        self.p_t = None
        scale = self.cycle_iterations / (-2 * np.log(0.001))

        def tau_function(i):
            """
            A function the generate the gumbel temperature.
            Args:
                i: An int the represent the current iteration number

            Returns: A temperature value.

            """
            if i < (self.cycle_iterations - 1):
                index = ((i + 1) % self.cycle_iterations) / scale
            else:
                index = (i % self.cycle_iterations) / scale

            x = tf.exp(-index)
            return self.minimal_temp + (self.maximal_temp - self.minimal_temp) * x

        self.tau_function = tau_function
        self.w_shape = None
        self.update_gumbel_param = True

    def enable_update(self):
        self.update_gumbel_param = True

    def disable_update(self):
        self.update_gumbel_param = False

    def build(self, tensor_shape: TensorShape,
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
        w_shape = kernel_functions.get_kernel(layer.weights).shape
        self.w_shape = w_shape

        ar_iter = layer.add_weight(
            name + gptq_constants.GPTQ_ITER,
            shape=(),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=False)

        auxvar_tensor = layer.add_weight(
            name + gptq_constants.AUXVAR,
            shape=[self.m, *self.w_shape],
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)
        auxvar_tensor.assign(_init_aux_var(self.w_shape, self.m))

        temp_tensor = layer.add_weight(
            name + gptq_constants.TEMP,
            shape=[1, *self.w_shape],
            initializer=tf.keras.initializers.Constant(self.maximal_temp),
            trainable=True)

        shift_tensor = layer.add_weight(name + gptq_constants.AUXSHIFT,
                                        shape=self.m,
                                        initializer=tf.keras.initializers.Constant(0.0),
                                        trainable=False)
        shift_tensor.assign(_init_shift_var(self.m))

        self.quantizer_parameters = {gptq_constants.AUXVAR: auxvar_tensor,
                                     gptq_constants.GPTQ_ITER: ar_iter,
                                     gptq_constants.AUXSHIFT: shift_tensor,
                                     gptq_constants.TEMP: temp_tensor}
        return self.quantizer_parameters

    def get_aux_variable(self) -> tf.Tensor:
        return self.quantizer_parameters[gptq_constants.AUXVAR]

    def get_trainable_parameters(self) -> List[tf.Tensor]:
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
        if not isinstance(other, GumbelRoundingBase):
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

    def update_iteration(self, training, ar_iter):
        if self.temperature_learning:
            self.tau = tf.minimum(tf.maximum(self.quantizer_parameters[gptq_constants.TEMP], self.minimal_temp),
                                  self.maximal_temp)
        else:
            self.tau = self.tau_function(ar_iter)
        if self.update_gumbel_param and training:

            if ar_iter % self.cycle_iterations == 0:
                self.quantizer_parameters[gptq_constants.TEMP].assign(
                    self.maximal_temp * np.ones(self.quantizer_parameters[gptq_constants.TEMP].shape))
            ar_iter.assign_add(1.0)
            self.g_t = sample_gumbel([self.m, *self.w_shape])

    def get_temperature_variable(self):
        return self.quantizer_parameters[gptq_constants.TEMP]

    def get_gumbel_probability(self):
        return self.p_t
