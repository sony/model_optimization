# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

import copy
from enum import Enum
from typing import List

class QuantizationMethod(Enum):
    """
    Method for quantization function selection:

    POWER_OF_TWO - Symmetric, uniform, threshold is power of two quantization.

    KMEANS - k-means quantization.

    LUT_QUANTIZER - quantization using a look up table.

    SYMMETRIC - Symmetric, uniform, quantization.

    UNIFORM - uniform quantization,

    """
    POWER_OF_TWO = 0
    KMEANS = 1
    LUT_QUANTIZER = 2
    SYMMETRIC = 3
    UNIFORM = 4




class OpQuantizationConfig:
    """
    OpQuantizationConfig is a class to configure the quantization parameters of an operator.
    """

    def __init__(self,
                 activation_quantization_method: QuantizationMethod,
                 weights_quantization_method: QuantizationMethod,
                 activation_n_bits: int,
                 weights_n_bits: int,
                 weights_per_channel_threshold: bool,
                 enable_weights_quantization: bool,
                 enable_activation_quantization: bool,
                 quantization_preserving: bool,
                 fixed_scale: float,
                 fixed_zero_point: int,
                 weights_multiplier_nbits: int  # If None - set 8 in hptq, o.w use it
                 ):
        """

        Args:
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            weights_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for weights quantization.
            activation_n_bits (int): Number of bits to quantize the activations.
            weights_n_bits (int): Number of bits to quantize the coefficients.
            weights_per_channel_threshold (bool): Whether to quantize the weights per-channel or not (per-tensor).
            enable_weights_quantization (bool): Whether to quantize the model weights or not.
            enable_activation_quantization (bool): Whether to quantize the model activations or not.
            quantization_preserving (bool): Whether quantization parameters should be the same for an operator's input and output.
            fixed_scale (float): Scale to use for an operator quantization parameters.
            fixed_zero_point (int): Zero-point to use for an operator quantization parameters.
            weights_multiplier_nbits (int): Number of bits to use when quantizing in look-up-table.
        """

        self.activation_quantization_method = activation_quantization_method
        self.weights_quantization_method = weights_quantization_method
        self.activation_n_bits = activation_n_bits
        self.weights_n_bits = weights_n_bits
        self.weights_per_channel_threshold = weights_per_channel_threshold
        self.enable_weights_quantization = enable_weights_quantization
        self.enable_activation_quantization = enable_activation_quantization
        self.quantization_preserving = quantization_preserving
        self.fixed_scale = fixed_scale
        self.fixed_zero_point = fixed_zero_point
        self.weights_multiplier_nbits = weights_multiplier_nbits

    def get_info(self):
        """

        Returns: Info about the quantization configuration as a dictionary.

        """
        return self.__dict__

    def clone_and_edit(self, **kwargs):
        """
        Clone the quantization config and edit some of its attributes.
        Args:
            **kwargs: Keyword arguments to edit the configuration to clone.

        Returns:
            Edited quantization configuration.
        """

        qc = copy.deepcopy(self)
        for k, v in kwargs.items():
            assert hasattr(qc,
                           k), f'Edit attributes is possible only for existing attributes in configuration, ' \
                               f'but {k} is not an attribute of {qc}'
            setattr(qc, k, v)
        return qc

    def __eq__(self, other):
        """
        Is this configuration equal to another object.
        Args:
            other: Object to compare.

        Returns:
            Whether this configuration is equal to another object or not.
        """
        if not isinstance(other, OpQuantizationConfig):
            return False
        return self.activation_quantization_method == other.activation_quantization_method and \
               self.weights_quantization_method == other.weights_quantization_method and \
               self.activation_n_bits == other.activation_n_bits and \
               self.weights_n_bits == other.weights_n_bits and \
               self.weights_per_channel_threshold == other.weights_per_channel_threshold and \
               self.enable_weights_quantization == other.enable_weights_quantization and \
               self.enable_activation_quantization == other.enable_activation_quantization


class QuantizationConfigOptions(object):
    """

    Wrap a set of quantization configurations to consider during the quantization
    of an operator.

    """
    def __init__(self,
                 quantization_config_list: List[OpQuantizationConfig],
                 base_config: OpQuantizationConfig = None):
        """

        Args:
            quantization_config_list (List[OpQuantizationConfig]): List of possible OpQuantizationConfig to gather.
            base_config (OpQuantizationConfig): Fallback OpQuantizationConfig to use when optimizing the model in a non mixed-precision manner.
        """

        assert isinstance(quantization_config_list,
                          list), f'QuantizationConfigOptions options list should be of type list, but is: ' \
                                 f'{type(quantization_config_list)}'
        assert len(quantization_config_list) > 0, f'Options list can not be empty'
        for cfg in quantization_config_list:
            assert isinstance(cfg, OpQuantizationConfig), f'Options should be a list of QuantizationConfig objects, ' \
                                                        f'but found an object type: {type(cfg)}'
        self.quantization_config_list = quantization_config_list
        if len(quantization_config_list)>1:
            assert base_config is not None, f'When quantization config options contains more than one configuration, a base_config must be passed for non-mixed-precision optimization process'
        self.base_config = base_config

    def __eq__(self, other):
        """
        Is this QCOptions equal to another object.
        Args:
            other: Object to compare.

        Returns:
            Whether this QCOptions equal to another object or not.
        """

        if not isinstance(other, QuantizationConfigOptions):
            return False
        if len(self.quantization_config_list) != len(other.quantization_config_list):
            return False
        for qc, other_qc in zip(self.quantization_config_list, other.quantization_config_list):
            if qc != other_qc:
                return False
        return True

    def clone_and_edit(self, **kwargs):
        qc_options = copy.deepcopy(self)
        for qc in qc_options.quantization_config_list:
            self.__edit_quantization_configuration(qc, kwargs)
        return qc_options

    def __edit_quantization_configuration(self, qc, kwargs):
        for k, v in kwargs.items():
            assert hasattr(qc,
                           k), f'Edit attributes is possible only for existing attributes in configuration, ' \
                               f'but {k} is not an attribute of {qc}'
            setattr(qc, k, v)

    def get_info(self):
        return {f'option {i}': cfg.get_info() for i, cfg in enumerate(self.quantization_config_list)}

