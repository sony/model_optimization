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


from typing import Callable

import numpy as np

from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig


##########################################
# Every node holds a quantization configuration
# for its weights quantization, and a different quantization
# configuration for its activation quantization configuration.
##########################################

class BaseNodeNodeQuantizationConfig(object):
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
            self.__dict__[attr_name] = attr_value

    def __repr__(self) -> str:
        """
        Returns: String to display a NodeQuantizationConfig object.
        """
        repr_str = ''
        for k, v in self.__dict__.items():
            repr_str += f'{k}: {v}\n'
        return repr_str


class NodeActivationQuantizationConfig(BaseNodeNodeQuantizationConfig):
    """
    Attributes for configuring the quantization of the activations of a node.
    """
    def __init__(self,
                 qc: QuantizationConfig,
                 activation_quantization_fn: Callable,
                 activation_quantization_params_fn: Callable,
                 activation_is_signed: bool = None
                 ):
        """

        Args:
            qc: QuantizationConfig to create the node's config from.
            activation_quantization_fn: Function to use when quantizing the node's activations.
            activation_quantization_params_fn: Function to use when computing the threshold for quantizing a node's activations.
            activation_is_signed: Signedness of the activation quantized range.
        """
        self.activation_quantization_fn = activation_quantization_fn
        self.activation_quantization_params_fn = activation_quantization_params_fn
        self.activation_is_signed = activation_is_signed
        self.activation_quantization_params = {}
        self.activation_threshold_method = qc.activation_threshold_method
        self.activation_quantization_method = qc.activation_quantization_method
        self.activation_n_bits = qc.activation_n_bits
        self.relu_unbound_correction = qc.relu_unbound_correction
        self.enable_activation_quantization = qc.enable_activation_quantization
        self.activation_channel_equalization = qc.activation_channel_equalization
        self.input_scaling = qc.input_scaling
        self.min_threshold = qc.min_threshold
        self.l_p_value = qc.l_p_value
        self.shift_negative_activation_correction = qc.shift_negative_activation_correction
        self.z_threshold = qc.z_threshold
        self.shift_negative_ratio = qc.shift_negative_ratio
        self.shift_negative_threshold_recalculation = qc.shift_negative_threshold_recalculation

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


class NodeWeightsQuantizationConfig(BaseNodeNodeQuantizationConfig):
    """
    Attributes for configuring the quantization of the weights of a node.
    """
    def __init__(self,
                 qc: QuantizationConfig,
                 weights_quantization_fn: Callable,
                 weights_quantization_params_fn: Callable,
                 weights_channels_axis: int):
        """

        Args:
            qc: QuantizationConfig to create the node's config from.
            weights_quantization_fn: Function to use when quantizing the node's weights.
            weights_quantization_params_fn:  Function to use when computing the threshold for quantizing a node's weights.
            weights_channels_axis: Axis to quantize a node's kernel when quantizing per-channel.
        """

        self.weights_quantization_fn = weights_quantization_fn
        self.weights_quantization_params_fn = weights_quantization_params_fn
        self.weights_channels_axis = weights_channels_axis
        self.weights_quantization_params = {}
        self.weights_threshold_method = qc.weights_threshold_method
        self.weights_quantization_method = qc.weights_quantization_method
        self.weights_n_bits = qc.weights_n_bits
        self.weights_bias_correction = qc.weights_bias_correction
        self.weights_per_channel_threshold = qc.weights_per_channel_threshold
        self.enable_weights_quantization = qc.enable_weights_quantization
        self.min_threshold = qc.min_threshold
        self.l_p_value = qc.l_p_value

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
        for param_name, param_value in weights_params.items():
            self.weights_quantization_params[param_name] = param_value

    def calculate_and_set_weights_params(self, tensor_data: np.ndarray) -> float:
        """
        Args:
            tensor_data: Tensor content as Numpy array.

        Returns:
            Recalculated weights quantization params from the kernel and channel axis.

        """

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


