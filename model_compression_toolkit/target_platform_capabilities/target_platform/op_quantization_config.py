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

import copy
from typing import List, Dict, Union, Any

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.logger import Logger


def clone_and_edit_object_params(obj: Any, **kwargs: Dict) -> Any:
    """
    Clones the given object and edit some of its parameters.

    Args:
        obj: An object to clone.
        **kwargs: Keyword arguments to edit in the cloned object.

    Returns:
        Edited copy of the given object.
    """

    obj_copy = copy.deepcopy(obj)
    for k, v in kwargs.items():
        assert hasattr(obj_copy,
                       k), f'Edit parameter is possible only for existing parameters in the given object, ' \
                           f'but {k} is not a parameter of {obj_copy}.'
        setattr(obj_copy, k, v)
    return obj_copy


class AttributeQuantizationConfig:
    """
    Hold the quantization configuration of a weight attribute of a layer.
    """
    def __init__(self,
                 weights_quantization_method: QuantizationMethod = QuantizationMethod.POWER_OF_TWO,
                 weights_n_bits: int = FLOAT_BITWIDTH,
                 weights_per_channel_threshold: bool = False,
                 enable_weights_quantization: bool = False,
                 lut_values_bitwidth: Union[int, None] = None,  # If None - set 8 in hptq, o.w use it
                 ):
        """
        Initializes an attribute quantization config.

        Args:
            weights_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for weights quantization.
            weights_n_bits (int): Number of bits to quantize the coefficients.
            weights_per_channel_threshold (bool): Whether to quantize the weights per-channel or not (per-tensor).
            enable_weights_quantization (bool): Whether to quantize the model weights or not.
            lut_values_bitwidth (int): Number of bits to use when quantizing in look-up-table.

        """

        self.weights_quantization_method = weights_quantization_method
        self.weights_n_bits = weights_n_bits
        self.weights_per_channel_threshold = weights_per_channel_threshold
        self.enable_weights_quantization = enable_weights_quantization
        self.lut_values_bitwidth = lut_values_bitwidth

    def clone_and_edit(self, **kwargs):
        """
        Clone the quantization config and edit some of its attributes.

        Args:
            **kwargs: Keyword arguments to edit the configuration to clone.

        Returns:
            Edited quantization configuration.
        """

        return clone_and_edit_object_params(self, **kwargs)

    def __eq__(self, other):
        """
        Is this configuration equal to another object.

        Args:
            other: Object to compare.

        Returns:

            Whether this configuration is equal to another object or not.
        """
        if not isinstance(other, AttributeQuantizationConfig):
            return False
        return self.weights_quantization_method == other.weights_quantization_method and \
            self.weights_n_bits == other.weights_n_bits and \
            self.weights_per_channel_threshold == other.weights_per_channel_threshold and \
            self.enable_weights_quantization == other.enable_weights_quantization and \
            self.lut_values_bitwidth == other.lut_values_bitwidth


class OpQuantizationConfig:
    """
    OpQuantizationConfig is a class to configure the quantization parameters of an operator.
    """

    def __init__(self,
                 default_weight_attr_config: AttributeQuantizationConfig,
                 attr_weights_configs_mapping: Dict[str, AttributeQuantizationConfig],
                 activation_quantization_method: QuantizationMethod,
                 activation_n_bits: int,
                 enable_activation_quantization: bool,
                 quantization_preserving: bool,
                 fixed_scale: float,
                 fixed_zero_point: int,
                 simd_size: int
                 ):
        """

        Args:
            default_weight_attr_config (AttributeQuantizationConfig): A default attribute quantization configuration for the operation.
            attr_weights_configs_mapping (Dict[str, AttributeQuantizationConfig]): A mapping between an op attribute name and its quantization configuration.
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            activation_n_bits (int): Number of bits to quantize the activations.
            enable_activation_quantization (bool): Whether to quantize the model activations or not.
            quantization_preserving (bool): Whether quantization parameters should be the same for an operator's input and output.
            fixed_scale (float): Scale to use for an operator quantization parameters.
            fixed_zero_point (int): Zero-point to use for an operator quantization parameters.
            simd_size (int): Per op integer representing the Single Instruction, Multiple Data (SIMD) width of an operator. It indicates the number of data elements that can be fetched and processed simultaneously in a single instruction.

        """

        self.default_weight_attr_config = default_weight_attr_config
        self.attr_weights_configs_mapping = attr_weights_configs_mapping

        self.activation_quantization_method = activation_quantization_method
        self.activation_n_bits = activation_n_bits
        self.enable_activation_quantization = enable_activation_quantization
        self.quantization_preserving = quantization_preserving
        self.fixed_scale = fixed_scale
        self.fixed_zero_point = fixed_zero_point
        self.simd_size = simd_size

    def get_info(self):
        """

        Returns: Info about the quantization configuration as a dictionary.

        """
        return self.__dict__

    def clone_and_edit(self, attr_to_edit: Dict[str, Dict[str, Any]] = {}, **kwargs):
        """
        Clone the quantization config and edit some of its attributes.
        Args:
            attr_to_edit: A mapping between attributes names to edit and their parameters that
            should be edited to a new value.
            **kwargs: Keyword arguments to edit the configuration to clone.

        Returns:
            Edited quantization configuration.
        """

        qc = clone_and_edit_object_params(self, **kwargs)

        # optionally: editing specific parameters in the config of specified attributes
        edited_attrs = copy.deepcopy(qc.attr_weights_configs_mapping)
        for attr_name, attr_cfg in qc.attr_weights_configs_mapping.items():
            if attr_name in attr_to_edit:
                edited_attrs[attr_name] = attr_cfg.clone_and_edit(**attr_to_edit[attr_name])

        qc.attr_weights_configs_mapping = edited_attrs

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
        return self.default_weight_attr_config == other.default_weight_attr_config and \
            self.attr_weights_configs_mapping == other.attr_weights_configs_mapping and \
            self.activation_quantization_method == other.activation_quantization_method and \
            self.activation_n_bits == other.activation_n_bits and \
            self.enable_activation_quantization == other.enable_activation_quantization and \
            self.simd_size == other.simd_size


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
                          list), f'\'QuantizationConfigOptions\' options list must be a list, but received: {type(quantization_config_list)}.'
        assert len(quantization_config_list) > 0, f'Options list can not be empty.'
        for cfg in quantization_config_list:
            assert isinstance(cfg, OpQuantizationConfig), f'Each option must be an instance of \'OpQuantizationConfig\', but found an object of type: {type(cfg)}.'
        self.quantization_config_list = quantization_config_list
        if len(quantization_config_list) > 1:
            assert base_config is not None, f'For multiple configurations, a \'base_config\' is required for non-mixed-precision optimization.'
            assert base_config in quantization_config_list, f"\'base_config\' must be included in the quantization config options list."
            self.base_config = base_config
        elif len(quantization_config_list) == 1:
            self.base_config = quantization_config_list[0]
        else:
            Logger.critical("\'QuantizationConfigOptions\' requires at least one \'OpQuantizationConfig\'; the provided list is empty.")

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

    def clone_and_edit_weight_attribute(self, attrs: List[str] = None, **kwargs):
        """
        Clones the quantization configurations and edits some of their attributes' parameters.

        Args:
            attrs: attributes names to clone their configurations. If None is provided, updating the configurations
                of all attributes in the operation attributes config mapping.
            **kwargs: Keyword arguments to edit in the attributes configuration.

        Returns:
            QuantizationConfigOptions with edited attributes configurations.

        """

        qc_options = copy.deepcopy(self)

        for qc in qc_options.quantization_config_list:
            if attrs is None:
                attrs_to_update = list(qc.attr_weights_configs_mapping.keys())
            else:
                if not isinstance(attrs, List):
                    Logger.critical(f"Expected a list of attributes but received {type(attrs)}.")
                attrs_to_update = attrs

            for attr in attrs_to_update:
                if qc.attr_weights_configs_mapping.get(attr) is None:
                    Logger.critical(f'Editing attributes is only possible for existing attributes in the configuration\'s '
                                    f'weights config mapping; {attr} does not exist in {qc}.')
                self.__edit_quantization_configuration(qc.attr_weights_configs_mapping[attr], kwargs)
        return qc_options

    def clone_and_map_weights_attr_keys(self, layer_attrs_mapping: Union[Dict[str, str], None]):
        """
       Clones the quantization configuration options and edits the keys in each configuration attributes config mapping,
       based on the given attributes names mapping.

        Args:
            layer_attrs_mapping: A mapping between attributes names.

        Returns:
            QuantizationConfigOptions with edited attributes names.

        """
        qc_options = copy.deepcopy(self)

        # Extract the list of existing quantization configurations from qc_options

        # Check if the base_config is already included in the quantization configuration list
        # If not, add base_config to the list of configurations to update
        cfgs_to_update = [cfg for cfg in qc_options.quantization_config_list]
        if not any(qc_options.base_config is cfg for cfg in cfgs_to_update):
            cfgs_to_update.append(qc_options.base_config)

        for qc in cfgs_to_update:
            if layer_attrs_mapping is None:
                qc.attr_weights_configs_mapping = {}
            else:
                new_attr_mapping = {}
                for attr in list(qc.attr_weights_configs_mapping.keys()):
                    new_key = layer_attrs_mapping.get(attr)
                    if new_key is None:
                        Logger.critical(f"Attribute \'{attr}\' does not exist in the provided attribute mapping.")

                    new_attr_mapping[new_key] = qc.attr_weights_configs_mapping.pop(attr)

                qc.attr_weights_configs_mapping.update(new_attr_mapping)

        return qc_options

    def __edit_quantization_configuration(self, qc, kwargs):
        for k, v in kwargs.items():
            assert hasattr(qc,
                           k), (f'Editing is only possible for existing attributes in the configuration; '
                                f'{k} is not an attribute of {qc}.')
            setattr(qc, k, v)

    def get_info(self):
        return {f'option {i}': cfg.get_info() for i, cfg in enumerate(self.quantization_config_list)}

