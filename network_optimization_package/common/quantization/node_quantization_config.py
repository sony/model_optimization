# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================
from typing import Callable

import numpy as np

from network_optimization_package.common.quantization.quantization_config import QuantizationConfig
from network_optimization_package.common.logger import Logger
from network_optimization_package.common.quantization.quantization_params_fn_selection import \
    get_activation_quantization_params_fn, get_weights_quantization_params_fn
from network_optimization_package.common.framework_info import FrameworkInfo


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


def create_node_activation_qc(qc: QuantizationConfig,
                              fw_info: FrameworkInfo,
                              use_min_max: bool) -> NodeActivationQuantizationConfig:
    """
    Create a activations quantization configuration from a QuantizationConfig object.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its weights/activations should be quantized)
        use_min_max: Whether the collected min/max statistics should be used when the threshold is computed or not.

    Returns:
        Activation quantization configuration of a node.
    """

    activation_quantization_fn = fw_info.activation_quantizer_mapping.get(qc.activation_quantization_method)
    if activation_quantization_fn is None:
        Logger.critical('Unknown quantization method for activations')

    activation_quantization_params_fn = get_activation_quantization_params_fn(qc.activation_quantization_method,
                                                                              qc.activation_threshold_method,
                                                                              use_min_max)

    return NodeActivationQuantizationConfig(qc,
                                            activation_quantization_fn,
                                            activation_quantization_params_fn)


def create_node_weights_qc(qc: QuantizationConfig,
                           fw_info: FrameworkInfo,
                           weight_channel_axis: int) -> NodeWeightsQuantizationConfig:
    """
    Create a weights quantization configuration from a QuantizationConfig object.

    Args:
        qc: QuantizationConfig to create the node's config from.
        fw_info: Information about the specific framework the node was created from (e.g., whether or not its weights/activations should be quantized)
        weight_channel_axis: Axis to quantize a node's kernel when quantizing per-channel.

    Returns:
        Weights quantization configuration of a node.
    """

    weights_quantization_fn = fw_info.weights_quantizer_mapping.get(qc.weights_quantization_method)

    if weights_quantization_fn is None:
        Logger.critical('Unknown quantization method for weights')

    weights_quantization_params_fn = get_weights_quantization_params_fn(qc.weights_quantization_method,
                                                                        qc.weights_threshold_method)

    return NodeWeightsQuantizationConfig(qc,
                                         weights_quantization_fn,
                                         weights_quantization_params_fn,
                                         weight_channel_axis)
