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


from typing import Callable, Any

import numpy as np

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn

from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig, \
    QuantizationErrorMethod
from model_compression_toolkit.target_platform_capabilities.target_platform import OpQuantizationConfig


##########################################
# Every node holds a quantization configuration
# for its weights and activations quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################


class BaseNodeQuantizationConfig(object):
    """
    Base class for node quantization configuration
    """

    def set_quant_config_attr(self, attr_name, attr_value):
        """
        Changes a BaseNodeQuantizationConfig's attribute.

        Args:
            attr_name: attribute name to change.
            attr_value: attribute value to change.

        """

        if hasattr(self, attr_name):
            setattr(self, attr_name, attr_value)

    def __repr__(self) -> str:
        """
        Returns: String to display a NodeQuantizationConfig object.
        """
        repr_str = ''
        for k, v in self.__dict__.items():
            repr_str += f'{k}: {v}\n'
        return repr_str


class NodeActivationQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Attributes for configuring the quantization of the activations of a node.
    """
    def __init__(self,
                 qc: QuantizationConfig,
                 op_cfg: OpQuantizationConfig,
                 activation_quantization_fn: Callable,
                 activation_quantization_params_fn: Callable
                 ):
        """

        Args:
            qc: QuantizationConfig to create the node's config from.
            op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
            activation_quantization_fn: Function to use when quantizing the node's activations.
            activation_quantization_params_fn: Function to use when computing the threshold for quantizing a node's activations.
        """

        self.activation_quantization_fn = activation_quantization_fn
        self.activation_quantization_params_fn = activation_quantization_params_fn
        self.activation_quantization_params = {}
        self.activation_quantization_method = op_cfg.activation_quantization_method
        self.activation_error_method = qc.activation_error_method
        self.activation_n_bits = op_cfg.activation_n_bits
        self.relu_bound_to_power_of_2 = qc.relu_bound_to_power_of_2
        self.enable_activation_quantization = op_cfg.enable_activation_quantization
        self.activation_channel_equalization = qc.activation_channel_equalization
        self.input_scaling = qc.input_scaling
        self.min_threshold = qc.min_threshold
        self.l_p_value = qc.l_p_value
        self.shift_negative_activation_correction = qc.shift_negative_activation_correction
        self.z_threshold = qc.z_threshold
        self.shift_negative_ratio = qc.shift_negative_ratio
        self.shift_negative_threshold_recalculation = qc.shift_negative_threshold_recalculation

    def quantize_node_output(self,
                             tensors: Any) -> Any:
        """

        Args:
            tensors: framework tensor/s

        Returns:
            Framework tensor/s after applying fake quantization.

        """
        fake_quant = self.activation_quantization_fn(self.activation_n_bits,
                                                     self.activation_quantization_params)

        if fake_quant is None:
            Logger.error('Layer is meant to be quantized but fake_quant function is None')  # pragma: no cover
        return fake_quant(tensors)

    @property
    def activation_error_method(self) -> QuantizationErrorMethod:
        """
        activation_error_method getter.
        """
        return self._activation_error_method

    @activation_error_method.setter
    def activation_error_method(self, value: QuantizationErrorMethod):
        """
        activation_error_method setter.

        Args:
            value: New activation_error_method to set to the node activation configuration.

        """
        self._activation_error_method = value
        self.activation_quantization_params_fn = get_activation_quantization_params_fn(activation_quantization_method=self.activation_quantization_method)

    def set_activation_quantization_fn(self, activation_quantization_fn: Callable):
        """
        Sets activation quantization function for the node.

        Args:
            activation_quantization_fn: Function for quantazing the activations.

        """
        self.activation_quantization_fn = activation_quantization_fn

    def set_activation_quantization_params_fn(self, activation_quantization_params_fn:Callable):
        """
        Sets activation params function for the node.

        Args:
            activation_quantization_params_fn: Function for calculating activation params.

        """
        self.activation_quantization_params_fn = activation_quantization_params_fn

    def set_activation_quantization_param(self,
                                          activation_params: dict):
        """
         Set a quantization parameter for the node's activation.

        Args:
            activation_params: Dictionary that contains weight quantization params.

        """
        assert self.enable_activation_quantization
        for param_name, param_value in activation_params.items():
            self.activation_quantization_params[param_name] = param_value

    def has_activation_quantization_params(self) -> bool:
        """

        Returns: Whether NodeQuantizationConfig has a activation quantization params or not.

        """
        return len(self.activation_quantization_params) > 0

    def no_quantization(self) -> bool:
        """
        Returns: Whether NodeQuantizationConfig does not have activation params.
        """
        return (not self.has_activation_quantization_params())

    def __eq__(self, other: Any) -> bool:
        """
        Compares the object to another object to find if they are equal.

        Args:
            other: An object to compare to.

        Returns: Whether the objects are identical or not.

        """
        if not isinstance(other, NodeActivationQuantizationConfig):
            return False

        return self.activation_quantization_fn == other.activation_quantization_fn and \
               self.activation_quantization_params_fn == other.activation_quantization_params_fn and \
               self.activation_error_method == other.activation_error_method and \
               self.activation_quantization_method == other.activation_quantization_method and \
               self.activation_n_bits == other.activation_n_bits and \
               self.enable_activation_quantization == other.enable_activation_quantization and \
               self.activation_channel_equalization == other.activation_channel_equalization and \
               self.input_scaling == other.input_scaling and \
               self.min_threshold == other.min_threshold and \
               self.l_p_value == other.l_p_value and \
               self.shift_negative_activation_correction == other.shift_negative_activation_correction and \
               self.z_threshold == other.z_threshold and \
               self.shift_negative_ratio == other.shift_negative_ratio and \
               self.shift_negative_threshold_recalculation == other.shift_negative_threshold_recalculation

    def __hash__(self):
        return hash((self.activation_quantization_fn,
                     self.activation_quantization_params_fn,
                     self.activation_error_method,
                     self.activation_quantization_method,
                     self.activation_n_bits,
                     self.enable_activation_quantization,
                     self.activation_channel_equalization,
                     self.input_scaling,
                     self.min_threshold,
                     self.l_p_value,
                     self.shift_negative_activation_correction,
                     self.z_threshold,
                     self.shift_negative_ratio,
                     self.shift_negative_threshold_recalculation))


class NodeWeightsQuantizationConfig(BaseNodeQuantizationConfig):
    """
    Attributes for configuring the quantization of the weights of a node.
    """
    def __init__(self,
                 qc: QuantizationConfig,
                 op_cfg: OpQuantizationConfig,
                 weights_quantization_fn: Callable,
                 weights_quantization_params_fn: Callable,
                 weights_channels_axis: int):
        """

        Args:
            qc: QuantizationConfig to create the node's config from.
            op_cfg: OpQuantizationConfig of the node with quantizers types to use when creating node quantization configuration.
            weights_quantization_fn: Function to use when quantizing the node's weights.
            weights_quantization_params_fn:  Function to use when computing the threshold for quantizing a node's weights.
            weights_channels_axis: Axis to quantize a node's kernel when quantizing per-channel.
        """

        self.weights_quantization_fn = weights_quantization_fn
        self.weights_quantization_params_fn = weights_quantization_params_fn
        self.weights_channels_axis = weights_channels_axis
        self.weights_quantization_params = {}
        self.weights_quantization_method = op_cfg.weights_quantization_method
        self.weights_error_method = qc.weights_error_method
        self.weights_n_bits = op_cfg.weights_n_bits
        self.weights_bias_correction = qc.weights_bias_correction
        self.weights_second_moment_correction = qc.weights_second_moment_correction
        self.weights_per_channel_threshold = op_cfg.weights_per_channel_threshold
        self.enable_weights_quantization = op_cfg.enable_weights_quantization
        self.min_threshold = qc.min_threshold
        self.l_p_value = qc.l_p_value


    @property
    def weights_error_method(self) -> QuantizationErrorMethod:
        """
        weights_error_method getter.
        """
        return self._weights_error_method

    @weights_error_method.setter
    def weights_error_method(self, value: QuantizationErrorMethod):
        """
        weights_error_method setter.

        Args:
            value: New weights_error_method to set to the node weights configuration.

        """
        self._weights_error_method = value
        self.weights_quantization_params_fn = get_weights_quantization_params_fn(weights_quantization_method=self.weights_quantization_method)


    def set_weights_quantization_fn(self, weights_quantization_fn: Callable):
        """
        Sets weights quantization function for the node.

        Args:
            weights_quantization_fn: Function for quantazing the weights.

        """
        self.weights_quantization_fn = weights_quantization_fn

    def set_weights_quantization_params_fn(self, weights_quantization_params_fn: Callable):
        """
        Sets weights params function for the node.

        Args:
            weights_quantization_params_fn: Function for calculating the weights params.

        """
        self.weights_quantization_params_fn = weights_quantization_params_fn

    def set_weights_quantization_param(self,
                                       weights_params: dict):
        """
         Set a quantization parameter for the node's weights.

        Args:
            weights_params: Dictionary that contains weight quantization params.

        """
        assert self.enable_weights_quantization
        for param_name, param_value in weights_params.items():
            self.weights_quantization_params[param_name] = param_value

    def calculate_and_set_weights_params(self, tensor_data: np.ndarray) -> float:
        """
        Args:
            tensor_data: Tensor content as Numpy array.

        Returns:
            Recalculated weights quantization params from the kernel and channel axis.

        """
        assert self.enable_weights_quantization
        if self.weights_quantization_params_fn is not None:
            self.set_weights_quantization_param(self.weights_quantization_params_fn(tensor_data,
                                                                                    p=self.l_p_value,
                                                                                    n_bits=self.weights_n_bits,
                                                                                    per_channel=self.weights_per_channel_threshold and self.weights_channels_axis is not None,
                                                                                    channel_axis=self.weights_channels_axis,
                                                                                    min_threshold=self.min_threshold))
        else:
            return self.set_weights_quantization_param({})

    def has_weights_quantization_params(self) -> bool:
        """

        Returns: Whether NodeQuantizationConfig has weights quantization params or not.

        """
        return len(self.weights_quantization_params) > 0

    def __eq__(self, other: Any) -> bool:
        """
        Compares the object to another object to find if they are equal.

        Args:
            other: An object to compare to.

        Returns: Whether the objects are identical or not.

        """
        if not isinstance(other, NodeWeightsQuantizationConfig):
            return False

        return self.weights_quantization_fn == other.weights_quantization_fn and \
               self.weights_quantization_params_fn == other.weights_quantization_params_fn and \
               self.weights_channels_axis == other.weights_channels_axis and \
               self.weights_error_method == other.weights_error_method and \
               self.weights_quantization_method == other.weights_quantization_method and \
               self.weights_n_bits == other.weights_n_bits and \
               self.weights_bias_correction == other.weights_bias_correction and \
               self.weights_second_moment_correction == other.weights_second_moment_correction and \
               self.weights_per_channel_threshold == other.weights_per_channel_threshold and \
               self.enable_weights_quantization == other.enable_weights_quantization and \
               self.min_threshold == other.min_threshold and \
               self.l_p_value == other.l_p_value

    def __hash__(self):
        return hash((self.weights_quantization_fn,
                     self.weights_quantization_params_fn,
                     self.weights_channels_axis,
                     self.weights_error_method,
                     self.weights_quantization_method,
                     self.weights_n_bits,
                     self.weights_bias_correction,
                     self.weights_second_moment_correction,
                     self.weights_per_channel_threshold,
                     self.enable_weights_quantization,
                     self.min_threshold,
                     self.l_p_value))
