# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import pprint

from typing import Dict, Any, Union, Tuple, List, Optional

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import FLOAT_BITWIDTH

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.constants import OPS_SET_LIST
from model_compression_toolkit.target_platform_capabilities.immutable import ImmutableClass
from model_compression_toolkit.target_platform_capabilities.target_platform.current_tp_model import \
    get_current_tp_model, _current_tp_model
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import clone_and_edit_object_params


class Signedness(Enum):
    """
    An enum for choosing the signedness of the quantization method:

    AUTO - Signedness decided automatically by quantization.
    SIGNED - Force signed quantization.
    UNSIGNED - Force unsigned quantization.
    """
    AUTO = 0
    SIGNED = 1
    UNSIGNED = 2


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
            return False  # pragma: no cover
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
                 supported_input_activation_n_bits: Union[int, Tuple[int]],
                 enable_activation_quantization: bool,
                 quantization_preserving: bool,
                 fixed_scale: float,
                 fixed_zero_point: int,
                 simd_size: int,
                 signedness: Signedness
                 ):
        """

        Args:
            default_weight_attr_config (AttributeQuantizationConfig): A default attribute quantization configuration for the operation.
            attr_weights_configs_mapping (Dict[str, AttributeQuantizationConfig]): A mapping between an op attribute name and its quantization configuration.
            activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
            activation_n_bits (int): Number of bits to quantize the activations.
            supported_input_activation_n_bits (int or Tuple[int]): Number of bits that operator accepts as input.
            enable_activation_quantization (bool): Whether to quantize the model activations or not.
            quantization_preserving (bool): Whether quantization parameters should be the same for an operator's input and output.
            fixed_scale (float): Scale to use for an operator quantization parameters.
            fixed_zero_point (int): Zero-point to use for an operator quantization parameters.
            simd_size (int): Per op integer representing the Single Instruction, Multiple Data (SIMD) width of an operator. It indicates the number of data elements that can be fetched and processed simultaneously in a single instruction.
            signedness (bool): Set activation quantization signedness.

        """

        self.default_weight_attr_config = default_weight_attr_config
        self.attr_weights_configs_mapping = attr_weights_configs_mapping

        self.activation_quantization_method = activation_quantization_method
        self.activation_n_bits = activation_n_bits
        if isinstance(supported_input_activation_n_bits, tuple):
            self.supported_input_activation_n_bits = supported_input_activation_n_bits
        elif isinstance(supported_input_activation_n_bits, int):
            self.supported_input_activation_n_bits = (supported_input_activation_n_bits,)
        else:
            Logger.critical(f"Supported_input_activation_n_bits only accepts int or tuple of ints, but got {type(supported_input_activation_n_bits)}")  # pragma: no cover
        self.enable_activation_quantization = enable_activation_quantization
        self.quantization_preserving = quantization_preserving
        self.fixed_scale = fixed_scale
        self.fixed_zero_point = fixed_zero_point
        self.signedness = signedness
        self.simd_size = simd_size

    def get_info(self):
        """

        Returns: Info about the quantization configuration as a dictionary.

        """
        return self.__dict__  # pragma: no cover

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
            return False  # pragma: no cover
        return self.default_weight_attr_config == other.default_weight_attr_config and \
            self.attr_weights_configs_mapping == other.attr_weights_configs_mapping and \
            self.activation_quantization_method == other.activation_quantization_method and \
            self.activation_n_bits == other.activation_n_bits and \
            self.supported_input_activation_n_bits == other.supported_input_activation_n_bits and \
            self.enable_activation_quantization == other.enable_activation_quantization and \
            self.signedness == other.signedness and \
            self.simd_size == other.simd_size

    @property
    def max_input_activation_n_bits(self) -> int:
        """
        Get maximum supported input bit-width.

        Returns: Maximum supported input bit-width.

        """
        return max(self.supported_input_activation_n_bits)


class QuantizationConfigOptions:
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
                          list), f"'QuantizationConfigOptions' options list must be a list, but received: {type(quantization_config_list)}."
        for cfg in quantization_config_list:
            assert isinstance(cfg, OpQuantizationConfig),\
                f"Each option must be an instance of 'OpQuantizationConfig', but found an object of type: {type(cfg)}."
        self.quantization_config_list = quantization_config_list
        if len(quantization_config_list) > 1:
            assert base_config is not None, \
                f"For multiple configurations, a 'base_config' is required for non-mixed-precision optimization."
            assert any([base_config is cfg for cfg in quantization_config_list]), \
                f"'base_config' must be included in the quantization config options list."
            # Enforce base_config to be a reference to an instance in quantization_config_list.
            self.base_config = base_config
        elif len(quantization_config_list) == 1:
            assert base_config is None or base_config == quantization_config_list[0], "'base_config' should be included in 'quantization_config_list'"
            # Set base_config to be a reference to the first instance in quantization_config_list.
            self.base_config = quantization_config_list[0]
        else:
            raise AssertionError("'QuantizationConfigOptions' requires at least one 'OpQuantizationConfig'. The provided list is empty.")

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
                if not isinstance(attrs, List):  # pragma: no cover
                    Logger.critical(f"Expected a list of attributes but received {type(attrs)}.")
                attrs_to_update = attrs

            for attr in attrs_to_update:
                if qc.attr_weights_configs_mapping.get(attr) is None:  # pragma: no cover
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
            # TODO: add test for this case
            cfgs_to_update.append(qc_options.base_config)

        for qc in cfgs_to_update:
            if layer_attrs_mapping is None:
                qc.attr_weights_configs_mapping = {}
            else:
                new_attr_mapping = {}
                for attr in list(qc.attr_weights_configs_mapping.keys()):
                    new_key = layer_attrs_mapping.get(attr)
                    if new_key is None:  # pragma: no cover
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


class TargetPlatformModelComponent:
    """
    Component of TargetPlatformModel (Fusing, OperatorsSet, etc.)
    """
    def __init__(self, name: str):
        """

        Args:
            name: Name of component.
        """
        self.name = name
        _current_tp_model.get().append_component(self)

    def get_info(self) -> Dict[str, Any]:
        """

        Returns: Get information about the component to display (return an empty dictionary.
        the actual component should fill it with info).

        """
        return {}


class OperatorsSetBase(TargetPlatformModelComponent):
    """
    Base class to represent a set of operators.
    """
    def __init__(self, name: str):
        """

        Args:
            name: Name of OperatorsSet.
        """
        super().__init__(name=name)


class OperatorsSet(OperatorsSetBase):
    def __init__(self,
                 name: str,
                 qc_options: QuantizationConfigOptions = None):
        """
        Set of operators that are represented by a unique label.

        Args:
            name (str): Set's label (must be unique in a TargetPlatformModel).
            qc_options (QuantizationConfigOptions): Configuration options to use for this set of operations.
        """

        super().__init__(name)
        self.qc_options = qc_options
        is_fusing_set = qc_options is None
        self.is_default = _current_tp_model.get().default_qco == self.qc_options or is_fusing_set


    def get_info(self) -> Dict[str,Any]:
        """

        Returns: Info about the set as a dictionary.

        """
        return {"name": self.name,
                "is_default_qc": self.is_default}


class OperatorSetConcat(OperatorsSetBase):
    """
    Concatenate a list of operator sets to treat them similarly in different places (like fusing).
    """
    def __init__(self, *opsets: OperatorsSet):
        """
        Group a list of operation sets.

        Args:
            *opsets (OperatorsSet): List of operator sets to group.
        """
        name = "_".join([a.name for a in opsets])
        super().__init__(name=name)
        self.op_set_list = opsets
        self.qc_options = None  # Concat have no qc options

    def get_info(self) -> Dict[str,Any]:
        """

        Returns: Info about the sets group as a dictionary.

        """
        return {"name": self.name,
                OPS_SET_LIST: [s.name for s in self.op_set_list]}


class Fusing(TargetPlatformModelComponent):
    """
     Fusing defines a list of operators that should be combined and treated as a single operator,
     hence no quantization is applied between them.
    """

    def __init__(self,
                 operator_groups_list: List[Union[OperatorsSet, OperatorSetConcat]],
                 name: str = None):
        """
        Args:
            operator_groups_list (List[Union[OperatorsSet, OperatorSetConcat]]): A list of operator groups, each being either an OperatorSetConcat or an OperatorsSet.
            name (str): The name for the Fusing instance. If not provided, it's generated from the operator groups' names.
        """
        assert isinstance(operator_groups_list,
                          list), f'List of operator groups should be of type list but is {type(operator_groups_list)}'
        assert len(operator_groups_list) >= 2, f'Fusing can not be created for a single operators group'

        # Generate a name from the operator groups if no name is provided
        if name is None:
            name = '_'.join([x.name for x in operator_groups_list])

        super().__init__(name)
        self.operator_groups_list = operator_groups_list

    def contains(self, other: Any) -> bool:
        """
        Determines if the current Fusing instance contains another Fusing instance.

        Args:
            other: The other Fusing instance to check against.

        Returns:
            A boolean indicating whether the other instance is contained within this one.
        """
        if not isinstance(other, Fusing):
            return False

        # Check for containment by comparing operator groups
        for i in range(len(self.operator_groups_list) - len(other.operator_groups_list) + 1):
            for j in range(len(other.operator_groups_list)):
                if self.operator_groups_list[i + j] != other.operator_groups_list[j] and not (
                        isinstance(self.operator_groups_list[i + j], OperatorSetConcat) and (
                        other.operator_groups_list[j] in self.operator_groups_list[i + j].op_set_list)):
                    break
            else:
                # If all checks pass, the other Fusing instance is contained
                return True
        # Other Fusing instance is not contained
        return False

    def get_info(self):
        """
        Retrieves information about the Fusing instance, including its name and the sequence of operator groups.

        Returns:
            A dictionary with the Fusing instance's name as the key and the sequence of operator groups as the value,
            or just the sequence of operator groups if no name is set.
        """
        if self.name is not None:
            return {self.name: ' -> '.join([x.name for x in self.operator_groups_list])}
        return ' -> '.join([x.name for x in self.operator_groups_list])


class TargetPlatformModel(ImmutableClass):
    """
    Represents the hardware configuration used for quantized model inference.

    This model defines:
    - The operators and their associated quantization configurations.
    - Fusing patterns, enabling multiple operators to be combined into a single operator
      for optimization during inference.
    - Versioning support through minor and patch versions for backward compatibility.

    Attributes:
        SCHEMA_VERSION (int): The schema version of the target platform model.
    """
    SCHEMA_VERSION = 1
    def __init__(self,
                 default_qco: QuantizationConfigOptions,
                 tpc_minor_version: Optional[int],
                 tpc_patch_version: Optional[int],
                 tpc_platform_type: Optional[str],
                 add_metadata: bool = True,
                 name="default_tp_model"):
        """

        Args:
            default_qco (QuantizationConfigOptions): Default QuantizationConfigOptions to use for operators that their QuantizationConfigOptions are not defined in the model.
            tpc_minor_version (Optional[int]): The minor version of the target platform capabilities.
            tpc_patch_version (Optional[int]): The patch version of the target platform capabilities.
            tpc_platform_type (Optional[str]): The platform type of the target platform capabilities.
            add_metadata (bool): Whether to add metadata to the model or not.
            name (str): Name of the model.

         Raises:
            AssertionError: If the provided `default_qco` does not contain exactly one quantization configuration.
        """

        super().__init__()
        self.tpc_minor_version = tpc_minor_version
        self.tpc_patch_version = tpc_patch_version
        self.tpc_platform_type = tpc_platform_type
        self.add_metadata = add_metadata
        self.name = name
        self.operator_set = []
        assert isinstance(default_qco, QuantizationConfigOptions), \
            "default_qco must be an instance of QuantizationConfigOptions"
        assert len(default_qco.quantization_config_list) == 1, \
            "Default QuantizationConfigOptions must contain exactly one option."

        self.default_qco = default_qco
        self.fusing_patterns = []
        self.is_simd_padding = False

    def get_config_options_by_operators_set(self,
                                            operators_set_name: str) -> QuantizationConfigOptions:
        """
        Get the QuantizationConfigOptions of a OperatorsSet by the OperatorsSet name.
        If the name is not in the model, the default QuantizationConfigOptions is returned.

        Args:
            operators_set_name: Name of OperatorsSet to get.

        Returns:
            QuantizationConfigOptions to use for ops in OperatorsSet named operators_set_name.
        """
        for op_set in self.operator_set:
            if operators_set_name == op_set.name:
                return op_set.qc_options
        return self.default_qco

    def get_default_op_quantization_config(self) -> OpQuantizationConfig:
        """

        Returns: The default OpQuantizationConfig of the TargetPlatformModel.

        """
        assert len(self.default_qco.quantization_config_list) == 1, \
            f'Default quantization configuration options must contain only one option,' \
            f' but found {len(get_current_tp_model().default_qco.quantization_config_list)} configurations.'
        return self.default_qco.quantization_config_list[0]

    def is_opset_in_model(self,
                          opset_name: str) -> bool:
        """
        Check whether an operators set is defined in the model or not.

        Args:
            opset_name: Operators set name to check.

        Returns:
            Whether an operators set is defined in the model or not.
        """
        return opset_name in [x.name for x in self.operator_set]

    def get_opset_by_name(self,
                          opset_name: str) -> OperatorsSetBase:
        """
        Get an OperatorsSet object from the model by its name.
        If name is not in the model - None is returned.

        Args:
            opset_name: OperatorsSet name to retrieve.

        Returns:
            OperatorsSet object with the name opset_name, or None if opset_name is not in the model.
        """

        opset_list = [x for x in self.operator_set if x.name == opset_name]
        assert len(opset_list) <= 1, f'Found more than one OperatorsSet in' \
                                     f' TargetPlatformModel with the name {opset_name}. ' \
                                     f'OperatorsSet name must be unique.'
        if len(opset_list) == 0:  # opset_name is not in the model.
            return None

        return opset_list[0]  # There's one opset with that name

    def append_component(self,
                         tp_model_component: TargetPlatformModelComponent):
        """
        Attach a TargetPlatformModel component to the model. Components can be for example:
        Fusing, OperatorsSet, etc.

        Args:
            tp_model_component: Component to attach to the model.

        """
        if isinstance(tp_model_component, Fusing):
            self.fusing_patterns.append(tp_model_component)
        elif isinstance(tp_model_component, OperatorsSetBase):
            self.operator_set.append(tp_model_component)
        else:  # pragma: no cover
            Logger.critical(f'Attempted to append an unrecognized TargetPlatformModelComponent of type: {type(tp_model_component)}.')

    def __enter__(self):
        """
        Start defining the TargetPlatformModel using 'with'.

        Returns: Initialized TargetPlatformModel object.

        """
        _current_tp_model.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Finish defining the TargetPlatformModel at the end of the 'with' clause.
        Returns the final and immutable TargetPlatformModel instance.
        """

        if exc_value is not None:
            print(exc_value, exc_value.args)
            raise exc_value
        self.__validate_model()  # Assert that model is valid.
        _current_tp_model.reset()
        self.initialized_done()  # Make model immutable.
        return self

    def __validate_model(self):
        """

        Assert model is valid.
        Model is invalid if, for example, it contains multiple operator sets with the same name,
        as their names should be unique.

        """
        opsets_names = [op.name for op in self.operator_set]
        if len(set(opsets_names)) != len(opsets_names):
            Logger.critical(f'Operator Sets must have unique names.')

    def get_default_config(self) -> OpQuantizationConfig:
        """

        Returns:

        """
        assert len(self.default_qco.quantization_config_list) == 1, \
            f'Default quantization configuration options must contain only one option,' \
            f' but found {len(self.default_qco.quantization_config_list)} configurations.'
        return self.default_qco.quantization_config_list[0]

    def get_info(self) -> Dict[str, Any]:
        """

        Returns: Dictionary that summarizes the TargetPlatformModel properties (for display purposes).

        """
        return {"Model name": self.name,
                "Default quantization config": self.get_default_config().get_info(),
                "Operators sets": [o.get_info() for o in self.operator_set],
                "Fusing patterns": [f.get_info() for f in self.fusing_patterns]
                }

    def show(self):
        """

        Display the TargetPlatformModel.

        """
        pprint.pprint(self.get_info(), sort_dicts=False)

    def set_simd_padding(self,
                         is_simd_padding: bool):
        """
        Set flag is_simd_padding to indicate whether this TP model defines
        that padding due to SIMD constrains occurs.

        Args:
            is_simd_padding: Whether this TP model defines that padding due to SIMD constrains occurs.

        """
        self.is_simd_padding = is_simd_padding

