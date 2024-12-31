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
import pprint

from enum import Enum
from typing import Dict, Any, Union, Tuple, List, Optional, Literal, Annotated
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.logger import Logger
from pydantic import BaseModel, Field,  ConfigDict, field_validator, model_validator


class OperatorSetNames(Enum):
    OPSET_CONV = "Conv"
    OPSET_DEPTHWISE_CONV = "DepthwiseConv2D"
    OPSET_CONV_TRANSPOSE = "ConvTranspose"
    OPSET_FULLY_CONNECTED = "FullyConnected"
    OPSET_CONCATENATE = "Concatenate"
    OPSET_STACK = "Stack"
    OPSET_UNSTACK = "Unstack"
    OPSET_GATHER = "Gather"
    OPSET_EXPAND = "Expend"
    OPSET_BATCH_NORM = "BatchNorm"
    OPSET_L2NORM = "L2Norm"
    OPSET_RELU = "ReLU"
    OPSET_RELU6 = "ReLU6"
    OPSET_LEAKY_RELU = "LEAKYReLU"
    OPSET_ELU = "Elu"
    OPSET_HARD_TANH = "HardTanh"
    OPSET_ADD = "Add"
    OPSET_SUB = "Sub"
    OPSET_MUL = "Mul"
    OPSET_DIV = "Div"
    OPSET_MIN = "Min"
    OPSET_MAX = "Max"
    OPSET_ADD_BIAS = "AddBias"
    OPSET_PRELU = "PReLU"
    OPSET_SWISH = "Swish"
    OPSET_SIGMOID = "Sigmoid"
    OPSET_SOFTMAX = "Softmax"
    OPSET_LOG_SOFTMAX = "LogSoftmax"
    OPSET_TANH = "Tanh"
    OPSET_GELU = "Gelu"
    OPSET_HARDSIGMOID = "HardSigmoid"
    OPSET_HARDSWISH = "HardSwish"
    OPSET_FLATTEN = "Flatten"
    OPSET_GET_ITEM = "GetItem"
    OPSET_RESHAPE = "Reshape"
    OPSET_UNSQUEEZE = "Unsqueeze"
    OPSET_SQUEEZE = "Squeeze"
    OPSET_PERMUTE = "Permute"
    OPSET_TRANSPOSE = "Transpose"
    OPSET_DROPOUT = "Dropout"
    OPSET_SPLIT_CHUNK = "SplitChunk"
    OPSET_MAXPOOL = "MaxPool"
    OPSET_SIZE = "Size"
    OPSET_SHAPE = "Shape"
    OPSET_EQUAL = "Equal"
    OPSET_ARGMAX = "ArgMax"
    OPSET_TOPK = "TopK"
    OPSET_FAKE_QUANT_WITH_MIN_MAX_VARS = "FakeQuantWithMinMaxVars"
    OPSET_COMBINED_NON_MAX_SUPPRESSION = "CombinedNonMaxSuppression"
    OPSET_CROPPING2D = "Cropping2D"
    OPSET_ZERO_PADDING2d = "ZeroPadding2D"
    OPSET_CAST = "Cast"
    OPSET_STRIDED_SLICE = "StridedSlice"
    OPSET_SSD_POST_PROCESS = "SSDPostProcess"

    @classmethod
    def get_values(cls):
        return [v.value for v in cls]


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


class AttributeQuantizationConfig(BaseModel):
    """
    Holds the quantization configuration of a weight attribute of a layer.

    Attributes:
        weights_quantization_method (QuantizationMethod): The method to use from QuantizationMethod for weights quantization.
        weights_n_bits (int): Number of bits to quantize the coefficients.
        weights_per_channel_threshold (bool): Indicates whether to quantize the weights per-channel or per-tensor.
        enable_weights_quantization (bool): Indicates whether to quantize the model weights or not.
        lut_values_bitwidth (Optional[int]): Number of bits to use when quantizing in a look-up table.
                                            If None, defaults to 8 in hptq; otherwise, it uses the provided value.
    """
    weights_quantization_method: QuantizationMethod = QuantizationMethod.POWER_OF_TWO
    weights_n_bits: int = FLOAT_BITWIDTH
    weights_per_channel_threshold: bool = False
    enable_weights_quantization: bool = False
    lut_values_bitwidth: Optional[int] = None

    model_config = ConfigDict(frozen=True)  # Makes the model immutable.

    @field_validator("weights_n_bits")
    def validate_weights_n_bits(cls, value):
        if value < 1:
            raise ValueError("weights_n_bits must be a positive integer.")
        return value

    @field_validator("lut_values_bitwidth", mode="before")
    def validate_lut_values_bitwidth(cls, value):
        if value is not None and not isinstance(value, int):
            raise ValueError("lut_values_bitwidth must be an integer or None.")
        return value

    def clone_and_edit(self, **kwargs) -> 'AttributeQuantizationConfig':
        """
        Clone the current AttributeQuantizationConfig and edit some of its attributes.

        Args:
            **kwargs: Keyword arguments representing the attributes to edit in the cloned instance.

        Returns:
            AttributeQuantizationConfig: A new instance of AttributeQuantizationConfig with updated attributes.
        """
        return self.model_copy(update=kwargs)


class OpQuantizationConfig(BaseModel):
    """
    OpQuantizationConfig is a class to configure the quantization parameters of an operator.

    Args:
        default_weight_attr_config (AttributeQuantizationConfig): A default attribute quantization configuration for the operation.
        attr_weights_configs_mapping (Dict[str, AttributeQuantizationConfig]): A mapping between an op attribute name and its quantization configuration.
        activation_quantization_method (QuantizationMethod): Which method to use from QuantizationMethod for activation quantization.
        activation_n_bits (int): Number of bits to quantize the activations.
        supported_input_activation_n_bits (Union[int, Tuple[int, ...]]): Number of bits that operator accepts as input.
        enable_activation_quantization (bool): Whether to quantize the model activations or not.
        quantization_preserving (bool): Whether quantization parameters should be the same for an operator's input and output.
        fixed_scale (Optional[float]): Scale to use for an operator quantization parameters.
        fixed_zero_point (Optional[int]): Zero-point to use for an operator quantization parameters.
        simd_size (Optional[int]): Per op integer representing the Single Instruction, Multiple Data (SIMD) width of an operator. It indicates the number of data elements that can be fetched and processed simultaneously in a single instruction.
        signedness (Signedness): Set activation quantization signedness.
    """
    default_weight_attr_config: AttributeQuantizationConfig
    attr_weights_configs_mapping: Dict[str, AttributeQuantizationConfig]
    activation_quantization_method: QuantizationMethod
    activation_n_bits: int
    supported_input_activation_n_bits: Union[int, Tuple[int, ...]]
    enable_activation_quantization: bool
    quantization_preserving: bool
    fixed_scale: Optional[float]
    fixed_zero_point: Optional[int]
    simd_size: Optional[int]
    signedness: Signedness

    model_config = ConfigDict(frozen=True)  # Makes the model immutable.

    @field_validator('supported_input_activation_n_bits', mode='before')
    def validate_supported_input_activation_n_bits(cls, v):
        """
        Validate and process the supported_input_activation_n_bits field.
        Converts an int to a tuple containing that int.
        Ensures that if a tuple is provided, all elements are ints.
        """

        # When loading from JSON, lists are returned. If the value is a list, convert it to a tuple.
        if isinstance(v, list):
            v = tuple(v)

        if isinstance(v, int):
            return (v,)
        elif isinstance(v, tuple):
            if all(isinstance(n, int) for n in v):
                return v
            else:
                Logger.critical(
                    f"All elements in supported_input_activation_n_bits must be integers, but got types {[type(n) for n in v]}"
                ) # pragma: no cover
        else:
            Logger.critical(
                f"supported_input_activation_n_bits only accepts int or tuple of ints, but got {type(v)}"
            )# pragma: no cover

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the quantization configuration.

        Returns:
            dict: Information about the quantization configuration as a dictionary.
        """
        return self.model_dump() # pragma: no cover

    def clone_and_edit(
        self,
        attr_to_edit: Dict[str, Dict[str, Any]] = {},
        **kwargs: Any
    ) -> 'OpQuantizationConfig':
        """
        Clone the quantization config and edit some of its attributes.

        Args:
            attr_to_edit (Dict[str, Dict[str, Any]]): A mapping between attribute names to edit and their parameters that
                                                     should be edited to a new value.
            **kwargs: Keyword arguments to edit the configuration to clone.

        Returns:
            OpQuantizationConfig: Edited quantization configuration.
        """
        # Clone and update top-level attributes
        updated_config = self.model_copy(update=kwargs)

        # Clone and update nested immutable dataclasses in `attr_weights_configs_mapping`
        updated_attr_mapping = {
            attr_name: (attr_cfg.clone_and_edit(**attr_to_edit[attr_name])
            if attr_name in attr_to_edit else attr_cfg)
            for attr_name, attr_cfg in updated_config.attr_weights_configs_mapping.items()
        }

        # Return a new instance with the updated attribute mapping
        return updated_config.model_copy(update={'attr_weights_configs_mapping': updated_attr_mapping})


class QuantizationConfigOptions(BaseModel):
    """
    QuantizationConfigOptions wraps a set of quantization configurations to consider during the quantization of an operator.

    Attributes:
        quantization_configurations (Tuple[OpQuantizationConfig]): Tuple of possible OpQuantizationConfig to gather.
        base_config (Optional[OpQuantizationConfig]): Fallback OpQuantizationConfig to use when optimizing the model in a non-mixed-precision manner.
    """
    quantization_configurations: Tuple[OpQuantizationConfig, ...]
    base_config: Optional[OpQuantizationConfig] = None

    # Pydantic v2 configuration for immutability
    model_config = ConfigDict(frozen=True)

    @model_validator(mode='before')
    def validate_and_set_base_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and set the base_config based on quantization_configurations.

        Args:
            values (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Modified input data with base_config set appropriately.
        """
        quantization_configurations = values.get('quantization_configurations', ())
        num_configs = len(quantization_configurations)
        base_config = values.get('base_config')


        if not isinstance(quantization_configurations, (tuple, list)):
            Logger.critical(
                f"'quantization_configurations' must be a list or tuple, but received: {type(quantization_configurations)}.") # pragma: no cover

        if num_configs > 1:
            if base_config is None:
                Logger.critical(
                    "For multiple configurations, a 'base_config' is required for non-mixed-precision optimization.") # pragma: no cover
            if base_config not in quantization_configurations:
                Logger.critical("'base_config' must be included in the quantization config options.") # pragma: no cover
        elif num_configs == 1:
            if base_config is None:
                # Automatically set base_config to the sole configuration
                values['base_config'] = quantization_configurations[0]
            elif base_config != quantization_configurations[0]:
                Logger.critical("'base_config' should be the same as the sole item in 'quantization_configurations'.") # pragma: no cover
        else:
            Logger.critical(
                "'QuantizationConfigOptions' requires at least one 'OpQuantizationConfig'. The provided configurations are empty.") # pragma: no cover

        # When loading from JSON, lists are returned. If the value is a list, convert it to a tuple.
        if isinstance(quantization_configurations, list):
            values['quantization_configurations'] = tuple(quantization_configurations)

        return values

    @model_validator(mode='after')
    def validate_after_initialization(cls, qco: 'QuantizationConfigOptions') -> Dict[str, Any]:
        """
        Perform validation after the model has been instantiated.

        Args:
            qco (QuantizationConfigOptions): The instantiated quantization config options.

        Returns:
            QuantizationConfigOptions: The validated model.
        """
        for cfg in qco.quantization_configurations:
            if not isinstance(cfg, OpQuantizationConfig):
                Logger.critical(
                    f"Each option must be an instance of 'OpQuantizationConfig', but found an object of type: {type(cfg)}.") # pragma: no cover
        return qco

    def clone_and_edit(self, **kwargs) -> 'QuantizationConfigOptions':
        """
        Clone the quantization configuration options and edit attributes in each configuration.

        Args:
            **kwargs: Keyword arguments to edit in each configuration.

        Returns:
            QuantizationConfigOptions: A new instance with updated configurations.
        """
        # Clone and update base_config
        updated_base_config = self.base_config.clone_and_edit(**kwargs) if self.base_config else None

        # Clone and update all configurations
        updated_configs = tuple(cfg.clone_and_edit(**kwargs) for cfg in self.quantization_configurations)

        return self.model_copy(update={
            'base_config': updated_base_config,
            'quantization_configurations': updated_configs
        })

    def clone_and_edit_weight_attribute(self, attrs: Optional[List[str]] = None, **kwargs) -> 'QuantizationConfigOptions':
        """
        Clones the quantization configurations and edits some of their attributes' parameters.

        Args:
            attrs (Optional[List[str]]): Attribute names to clone and edit their configurations. If None, updates all attributes.
            **kwargs: Keyword arguments to edit in the attributes configuration.

        Returns:
            QuantizationConfigOptions: A new instance with edited attributes configurations.
        """
        updated_base_config = self.base_config
        updated_configs = []

        for qc in self.quantization_configurations:
            if attrs is None:
                attrs_to_update = list(qc.attr_weights_configs_mapping.keys())
            else:
                attrs_to_update = attrs

            # Ensure all attributes exist in the config
            for attr in attrs_to_update:
                if attr not in qc.attr_weights_configs_mapping:
                    Logger.critical(f"Attribute '{attr}' does not exist in {qc}.") # pragma: no cover

            # Update the specified attributes
            updated_attr_mapping = {
                attr: qc.attr_weights_configs_mapping[attr].clone_and_edit(**kwargs)
                for attr in attrs_to_update
            }

            # If the current config is the base_config, update it accordingly
            if qc == self.base_config:
                updated_base_config = qc.clone_and_edit(attr_weights_configs_mapping=updated_attr_mapping)

            # Update the current config with the new attribute mappings
            updated_cfg = qc.clone_and_edit(attr_weights_configs_mapping=updated_attr_mapping)
            updated_configs.append(updated_cfg)

        return self.model_copy(update={
            'base_config': updated_base_config,
            'quantization_configurations': tuple(updated_configs)
        })

    def clone_and_map_weights_attr_keys(self, layer_attrs_mapping: Optional[Dict[str, str]] = None) -> 'QuantizationConfigOptions':
        """
        Clones the quantization configurations and updates keys in attribute config mappings.

        Args:
            layer_attrs_mapping (Optional[Dict[str, str]]): A mapping between attribute names.

        Returns:
            QuantizationConfigOptions: A new instance with updated attribute keys.
        """
        new_base_config = self.base_config
        updated_configs = []

        for qc in self.quantization_configurations:
            if layer_attrs_mapping is None:
                new_attr_mapping = qc.attr_weights_configs_mapping
            else:
                new_attr_mapping = {
                    layer_attrs_mapping.get(attr, attr): cfg
                    for attr, cfg in qc.attr_weights_configs_mapping.items()
                }

            # If the current config is the base_config, update it accordingly
            if qc == self.base_config:
                new_base_config = qc.clone_and_edit(attr_weights_configs_mapping=new_attr_mapping)

            # Update the current config with the new attribute mappings
            updated_cfg = qc.clone_and_edit(attr_weights_configs_mapping=new_attr_mapping)
            updated_configs.append(updated_cfg)

        return self.model_copy(update={
            'base_config': new_base_config,
            'quantization_configurations': tuple(updated_configs)
        })

    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about each quantization configuration option.

        Returns:
            dict: Information about the quantization configuration options as a dictionary.
        """
        return {f'option_{i}': cfg.get_info() for i, cfg in enumerate(self.quantization_configurations)}


class TargetPlatformModelComponent(BaseModel):
    """
    Component of TargetPlatformModel (Fusing, OperatorsSet, etc.).
    """
    model_config = ConfigDict(frozen=True)


class OperatorsSetBase(TargetPlatformModelComponent):
    """
    Base class to represent a set of a target platform model component of operator set types.
    Inherits from TargetPlatformModelComponent.
    """
    pass


class OperatorsSet(OperatorsSetBase):
    """
    Set of operators that are represented by a unique label.

    Attributes:
        name (Union[str, OperatorSetNames]): The set's label (must be unique within a TargetPlatformModel).
        qc_options (Optional[QuantizationConfigOptions]): Configuration options to use for this set of operations.
            If None, it represents a fusing set.
    """
    name: Union[str, OperatorSetNames]
    qc_options: Optional[QuantizationConfigOptions] = None
    type: Literal["OperatorsSet"] = "OperatorsSet"

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='after')
    def validate_after_initialization(cls, op_set: 'OperatorsSet') -> 'TargetPlatformModel':
        """
        Perform validation after the model has been instantiated.

        Args:
            op_set (OperatorsSet): The instantiated fusing.

        Returns:
            OperatorSet: The validated model.
        """
        if op_set.type != 'OperatorsSet':
            Logger.critical("'type' field must not change from default value 'OperatorsSet'.")  # pragma: no cover

        return op_set

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the set as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the set name.
        """
        return {"name": self.name}


class OperatorSetConcat(OperatorsSetBase):
    """
    Concatenate a tuple of operator sets to treat them similarly in different places (like fusing).

    Attributes:
        operators_set (Tuple[OperatorsSet, ...]): Tuple of operator sets to group.
        qc_options (None): Configuration options for the set, always None for concatenated sets.
        name (str): Concatenated name generated from the names of the operator sets.
    """
    operators_set: Tuple[OperatorsSet, ...]
    qc_options: Optional[QuantizationConfigOptions] = None  # Always None for concatenated sets
    name: Optional[str] = None  # Will be set in the validator
    type: Literal["OperatorSetConcat"] = "OperatorSetConcat"

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='before')
    def validate_and_set_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the input and set the concatenated name based on the operators_set.

        Args:
            values (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Modified input data with 'name' set.
        """
        operators_set = values.get('operators_set', ())

        if values.get('name') is None:
            # Generate the concatenated name from the operator sets
            concatenated_name = "_".join([
                op.name.value if isinstance(op.name, OperatorSetNames) else op.name
                for op in operators_set
            ])
            values['name'] = concatenated_name

        # Ensure qc_options is None
        values['qc_options'] = None

        return values

    @model_validator(mode='after')
    def validate_after_initialization(cls, op_set_concat: 'OperatorSetConcat') -> 'TargetPlatformModel':
        """
        Perform validation after the model has been instantiated.

        Args:
            op_set_concat (OperatorSetConcat): The instantiated fusing.

        Returns:
            OperatorSetConcat: The validated model.
        """
        if not op_set_concat.operators_set:
            Logger.critical("OperatorSetConcat requires at least one OperatorsSet in 'operators_set'.") # pragma: no cover

        if op_set_concat.type != 'OperatorSetConcat':
            Logger.critical("'type' field must not change from default value 'OperatorSetConcat'.")# pragma: no cover

        return op_set_concat

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the concatenated operator sets as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the concatenated name.
        """
        return {"name": self.name, "operators_set": [op.get_info() for op in self.operators_set]}


class Fusing(TargetPlatformModelComponent):
    """
    Fusing defines a tuple of operators that should be combined and treated as a single operator,
    hence no quantization is applied between them.

    Attributes:
        operator_groups (Tuple[Union[OperatorsSet, OperatorSetConcat]]): A tuple of operator groups,
                                                                          each being either an OperatorSetConcat or an OperatorsSet.
        name (str): The name for the Fusing instance. If not provided, it is generated from the operator groups' names.
    """
    operator_groups: Tuple[Annotated[Union[OperatorsSet, OperatorSetConcat], Field(discriminator='type')],...]
    name: Optional[str] = None  # Will be set in the validator

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='before')
    def validate_and_set_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the operator_groups and set the name by concatenating operator group names.

        Args:
            values (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Modified input data with 'name' set.
        """
        operator_groups = values.get('operator_groups', ())

        # When loading from JSON, lists are returned. If the value is a list, convert it to a tuple.
        if isinstance(operator_groups, list):
            values['operator_groups'] = tuple(operator_groups)

        if values.get("name") is None:
            # Generate the concatenated name from the operator groups
            concatenated_name = "_".join([
                op.name.value if isinstance(op.name, OperatorSetNames) else op.name
                for op in operator_groups
            ])
            values['name'] = concatenated_name

        return values

    @model_validator(mode='after')
    def validate_after_initialization(cls, fusing: 'Fusing') -> 'TargetPlatformModel':
        """
        Perform validation after the model has been instantiated.

        Args:
            fusing (Fusing): The instantiated fusing.

        Returns:
            Fusing: The validated model.
        """
        # Validate operator_groups is a tuple
        if not isinstance(fusing.operator_groups, tuple):
            Logger.critical(
                f"Operator groups should be of type 'tuple' but is {type(fusing.operator_groups)}."
            )  # pragma: no cover

        # Validate that there are at least two operator groups
        if len(fusing.operator_groups) < 2:
            Logger.critical("Fusing cannot be created for a single operator.")  # pragma: no cover

        return fusing

    def contains(self, other: Any) -> bool:
        """
        Determines if the current Fusing instance contains another Fusing instance.

        Args:
            other (Any): The other Fusing instance to check against.

        Returns:
            bool: True if the other Fusing instance is contained within this one, False otherwise.
        """
        if not isinstance(other, Fusing):
            return False

        # Check for containment by comparing operator groups
        for i in range(len(self.operator_groups) - len(other.operator_groups) + 1):
            for j in range(len(other.operator_groups)):
                if self.operator_groups[i + j] != other.operator_groups[j] and not (
                        isinstance(self.operator_groups[i + j], OperatorSetConcat) and (
                        other.operator_groups[j] in self.operator_groups[i + j].operators_set)):
                    break
            else:
                # If all checks pass, the other Fusing instance is contained
                return True
        # Other Fusing instance is not contained
        return False

    def get_info(self) -> Union[Dict[str, str], str]:
        """
        Retrieves information about the Fusing instance, including its name and the sequence of operator groups.

        Returns:
            Union[Dict[str, str], str]: A dictionary with the Fusing instance's name as the key
                                        and the sequence of operator groups as the value,
                                        or just the sequence of operator groups if no name is set.
        """
        if self.name is not None:
            return {
                self.name: ' -> '.join([
                    x.name.value if isinstance(x.name, OperatorSetNames) else x.name
                    for x in self.operator_groups
                ])
            }
        return ' -> '.join([
            x.name.value if isinstance(x.name, OperatorSetNames) else x.name
            for x in self.operator_groups
        ])

class TargetPlatformModel(BaseModel):
    """
    Represents the hardware configuration used for quantized model inference.

    Attributes:
        default_qco (QuantizationConfigOptions): Default quantization configuration options for the model.
        tpc_minor_version (Optional[int]): Minor version of the Target Platform Configuration.
        tpc_patch_version (Optional[int]): Patch version of the Target Platform Configuration.
        tpc_platform_type (Optional[str]): Type of the platform for the Target Platform Configuration.
        add_metadata (bool): Flag to determine if metadata should be added.
        name (str): Name of the Target Platform Model.
        operator_set (Tuple[OperatorsSetBase, ...]): Tuple of operator sets within the model.
        fusing_patterns (Tuple[Fusing, ...]): Tuple of fusing patterns for the model.
        is_simd_padding (bool): Indicates if SIMD padding is applied.
        SCHEMA_VERSION (int): Version of the schema for the Target Platform Model.
    """
    default_qco: QuantizationConfigOptions
    tpc_minor_version: Optional[int] = None
    tpc_patch_version: Optional[int] = None
    tpc_platform_type: Optional[str] = None
    add_metadata: bool = True
    name: Optional[str] = "default_tp_model"
    operator_set: Optional[Tuple[OperatorsSet, ...]] = None
    fusing_patterns: Optional[Tuple[Fusing, ...]] = None
    is_simd_padding: bool = False

    SCHEMA_VERSION: int = 1

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='after')
    def validate_after_initialization(cls, tp_model: 'TargetPlatformModel') -> 'TargetPlatformModel':
        """
        Perform validation after the model has been instantiated.

        Args:
            tp_model (TargetPlatformModel): The instantiated target platform model.

        Returns:
            TargetPlatformModel: The validated model.
        """
        # Validate `default_qco`
        default_qco = tp_model.default_qco
        if not isinstance(default_qco, QuantizationConfigOptions):
            Logger.critical("'default_qco' must be an instance of QuantizationConfigOptions.") # pragma: no cover

        if len(default_qco.quantization_configurations) != 1:
            Logger.critical("Default QuantizationConfigOptions must contain exactly one option.") # pragma: no cover

        # Validate `operator_set` uniqueness
        operator_set = tp_model.operator_set
        if operator_set is not None:
            opsets_names = [
                op.name.value if isinstance(op, OperatorSetNames) else op.name
                for op in operator_set
            ] if operator_set else []
            if len(set(opsets_names)) != len(opsets_names):
                Logger.critical("Operator Sets must have unique names.") # pragma: no cover

        return tp_model

    def get_info(self) -> Dict[str, Any]:
        """
        Get a dictionary summarizing the TargetPlatformModel properties.

        Returns:
            Dict[str, Any]: Summary of the TargetPlatformModel properties.
        """
        return {
            "Model name": self.name,
            "Operators sets": [o.get_info() for o in self.operator_set] if self.operator_set else [],
            "Fusing patterns": [f.get_info() for f in self.fusing_patterns] if self.fusing_patterns else [],
        }

    def show(self):
        """
        Display the TargetPlatformModel.
        """
        pprint.pprint(self.get_info(), sort_dicts=False)