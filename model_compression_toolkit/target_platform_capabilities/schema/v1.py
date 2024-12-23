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

from dataclasses import replace, dataclass, asdict, field
from enum import Enum
from typing import Dict, Any, Union, Tuple, List, Optional
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.constants import OPS_SET_LIST

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
    OPSET_RELU = "ReLU"
    OPSET_RELU6 = "ReLU6"
    OPSET_LEAKY_RELU = "LEAKYReLU"
    OPSET_HARD_TANH = "HardTanh"
    OPSET_ADD = "Add"
    OPSET_SUB = "Sub"
    OPSET_MUL = "Mul"
    OPSET_DIV = "Div"
    OPSET_MIN = "Min"
    OPSET_MAX = "Max"
    OPSET_PRELU = "PReLU"
    OPSET_SWISH = "Swish"
    OPSET_SIGMOID = "Sigmoid"
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
    OPSET_SPLIT = "Split"
    OPSET_CHUNK = "Chunk"
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


@dataclass(frozen=True)
class AttributeQuantizationConfig:
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

    def __post_init__(self):
        """
        Post-initialization processing for input validation.

        Raises:
            Logger critical if attributes are of incorrect type or have invalid values.
        """
        if not isinstance(self.weights_n_bits, int) or self.weights_n_bits < 1:
            Logger.critical("weights_n_bits must be a positive integer.") # pragma: no cover
        if not isinstance(self.enable_weights_quantization, bool):
            Logger.critical("enable_weights_quantization must be a boolean.") # pragma: no cover
        if self.lut_values_bitwidth is not None and not isinstance(self.lut_values_bitwidth, int):
            Logger.critical("lut_values_bitwidth must be an integer or None.") # pragma: no cover

    def clone_and_edit(self, **kwargs) -> 'AttributeQuantizationConfig':
        """
        Clone the current AttributeQuantizationConfig and edit some of its attributes.

        Args:
            **kwargs: Keyword arguments representing the attributes to edit in the cloned instance.

        Returns:
            AttributeQuantizationConfig: A new instance of AttributeQuantizationConfig with updated attributes.
        """
        return replace(self, **kwargs)


@dataclass(frozen=True)
class OpQuantizationConfig:
    """
    OpQuantizationConfig is a class to configure the quantization parameters of an operator.

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
    default_weight_attr_config: AttributeQuantizationConfig
    attr_weights_configs_mapping: Dict[str, AttributeQuantizationConfig]
    activation_quantization_method: QuantizationMethod
    activation_n_bits: int
    supported_input_activation_n_bits: Union[int, Tuple[int]]
    enable_activation_quantization: bool
    quantization_preserving: bool
    fixed_scale: float
    fixed_zero_point: int
    simd_size: int
    signedness: Signedness

    def __post_init__(self):
        """
        Post-initialization processing for input validation.

        Raises:
            Logger critical if supported_input_activation_n_bits is not an int or a tuple of ints.
        """
        if isinstance(self.supported_input_activation_n_bits, int):
            object.__setattr__(self, 'supported_input_activation_n_bits', (self.supported_input_activation_n_bits,))
        elif not isinstance(self.supported_input_activation_n_bits, tuple):
            Logger.critical(
                f"Supported_input_activation_n_bits only accepts int or tuple of ints, but got {type(self.supported_input_activation_n_bits)}")  # pragma: no cover

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the quantization configuration.

        Returns:
            dict: Information about the quantization configuration as a dictionary.
        """
        return asdict(self)  # pragma: no cover

    def clone_and_edit(self, attr_to_edit: Dict[str, Dict[str, Any]] = {}, **kwargs) -> 'OpQuantizationConfig':
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
        updated_config = replace(self, **kwargs)

        # Clone and update nested immutable dataclasses in `attr_weights_configs_mapping`
        updated_attr_mapping = {
            attr_name: (attr_cfg.clone_and_edit(**attr_to_edit[attr_name])
                        if attr_name in attr_to_edit else attr_cfg)
            for attr_name, attr_cfg in updated_config.attr_weights_configs_mapping.items()
        }

        # Return a new instance with the updated attribute mapping
        return replace(updated_config, attr_weights_configs_mapping=updated_attr_mapping)


@dataclass(frozen=True)
class QuantizationConfigOptions:
    """
    QuantizationConfigOptions wraps a set of quantization configurations to consider during the quantization of an operator.

    Attributes:
        quantization_configurations (Tuple[OpQuantizationConfig]): Tuple of possible OpQuantizationConfig to gather.
        base_config (Optional[OpQuantizationConfig]): Fallback OpQuantizationConfig to use when optimizing the model in a non-mixed-precision manner.
    """
    quantization_configurations: Tuple[OpQuantizationConfig]
    base_config: Optional[OpQuantizationConfig] = None

    def __post_init__(self):
        """
        Post-initialization processing for input validation.

        Raises:
            Logger critical if quantization_configurations is not a tuple, contains invalid elements, or if base_config is not set correctly.
        """
        # Validate `quantization_configurations`
        if not isinstance(self.quantization_configurations, tuple):
            Logger.critical(
                f"'quantization_configurations' must be a tuple, but received: {type(self.quantization_configurations)}.") # pragma: no cover
        for cfg in self.quantization_configurations:
            if not isinstance(cfg, OpQuantizationConfig):
                Logger.critical(
                    f"Each option must be an instance of 'OpQuantizationConfig', but found an object of type: {type(cfg)}.") # pragma: no cover

        # Handle base_config
        if len(self.quantization_configurations) > 1:
            if self.base_config is None:
                Logger.critical(f"For multiple configurations, a 'base_config' is required for non-mixed-precision optimization.") # pragma: no cover
            if not any(self.base_config == cfg for cfg in self.quantization_configurations):
                Logger.critical(f"'base_config' must be included in the quantization config options.") # pragma: no cover
        elif len(self.quantization_configurations) == 1:
            if self.base_config is None:
                object.__setattr__(self, 'base_config', self.quantization_configurations[0])
            elif self.base_config != self.quantization_configurations[0]:
                Logger.critical(
                    "'base_config' should be the same as the sole item in 'quantization_configurations'.") # pragma: no cover

        elif len(self.quantization_configurations) == 0:
            Logger.critical("'QuantizationConfigOptions' requires at least one 'OpQuantizationConfig'. The provided configurations is empty.") # pragma: no cover

    def clone_and_edit(self, **kwargs) -> 'QuantizationConfigOptions':
        """
        Clone the quantization configuration options and edit attributes in each configuration.

        Args:
            **kwargs: Keyword arguments to edit in each configuration.

        Returns:
            A new instance of QuantizationConfigOptions with updated configurations.
        """
        updated_base_config = replace(self.base_config, **kwargs)
        updated_configs = [
            replace(cfg, **kwargs) for cfg in self.quantization_configurations
        ]
        return replace(self, base_config=updated_base_config, quantization_configurations=tuple(updated_configs))

    def clone_and_edit_weight_attribute(self, attrs: List[str] = None, **kwargs) -> 'QuantizationConfigOptions':
        """
        Clones the quantization configurations and edits some of their attributes' parameters.

        Args:
            attrs (List[str]): Attributes names to clone and edit their configurations. If None, updates all attributes.
            **kwargs: Keyword arguments to edit in the attributes configuration.

        Returns:
            QuantizationConfigOptions: A new instance of QuantizationConfigOptions with edited attributes configurations.
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
                    Logger.critical(f"{attr} does not exist in {qc}.") # pragma: no cover
            updated_attr_mapping = {
                attr: qc.attr_weights_configs_mapping[attr].clone_and_edit(**kwargs)
                for attr in attrs_to_update
            }
            if qc == updated_base_config:
                updated_base_config = replace(updated_base_config, attr_weights_configs_mapping=updated_attr_mapping)
            updated_configs.append(replace(qc, attr_weights_configs_mapping=updated_attr_mapping))
        return replace(self, base_config=updated_base_config, quantization_configurations=tuple(updated_configs))

    def clone_and_map_weights_attr_keys(self, layer_attrs_mapping: Optional[Dict[str, str]]) -> 'QuantizationConfigOptions':
        """
        Clones the quantization configurations and updates keys in attribute config mappings.

        Args:
            layer_attrs_mapping (Optional[Dict[str, str]]): A mapping between attribute names.

        Returns:
            QuantizationConfigOptions: A new instance of QuantizationConfigOptions with updated attribute keys.
        """
        updated_configs = []
        new_base_config = self.base_config
        for qc in self.quantization_configurations:
            if layer_attrs_mapping is None:
                new_attr_mapping = {}
            else:
                new_attr_mapping = {
                    layer_attrs_mapping.get(attr, attr): cfg
                    for attr, cfg in qc.attr_weights_configs_mapping.items()
                }
            if qc == self.base_config:
                new_base_config = replace(qc, attr_weights_configs_mapping=new_attr_mapping)
            updated_configs.append(replace(qc, attr_weights_configs_mapping=new_attr_mapping))
        return replace(self, base_config=new_base_config, quantization_configurations=tuple(updated_configs))

    def get_info(self) -> Dict[str, Any]:
        """
        Get detailed information about each quantization configuration option.

        Returns:
            dict: Information about the quantization configuration options as a dictionary.
        """
        return {f'option {i}': cfg.get_info() for i, cfg in enumerate(self.quantization_configurations)}


@dataclass(frozen=True)
class TargetPlatformModelComponent:
    """
    Component of TargetPlatformModel (Fusing, OperatorsSet, etc.).
    """
    pass


@dataclass(frozen=True)
class OperatorsSetBase(TargetPlatformModelComponent):
    """
    Base class to represent a set of a target platform model component of operator set types.
    Inherits from TargetPlatformModelComponent.
    """
    pass


@dataclass(frozen=True)
class OperatorsSet(OperatorsSetBase):
    """
    Set of operators that are represented by a unique label.

    Attributes:
        name (str): The set's label (must be unique within a TargetPlatformModel).
        qc_options (QuantizationConfigOptions): Configuration options to use for this set of operations.
                                                If None, it represents a fusing set.
        is_default (bool): Indicates whether this set is the default quantization configuration
                           for the TargetPlatformModel or a fusing set.
    """
    name: Union[str, OperatorSetNames]
    qc_options: QuantizationConfigOptions = None

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the set as a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the set name and
                            whether it is the default quantization configuration.
        """
        return {"name": self.name}


@dataclass(frozen=True)
class OperatorSetConcat(OperatorsSetBase):
    """
    Concatenate a tuple of operator sets to treat them similarly in different places (like fusing).

    Attributes:
        operators_set (Tuple[OperatorsSet]): Tuple of operator sets to group.
        qc_options (None): Configuration options for the set, always None for concatenated sets.
        name (str): Concatenated name generated from the names of the operator sets.
    """
    operators_set: Tuple[OperatorsSet]
    qc_options: None = field(default=None, init=False)

    def __post_init__(self):
        """
        Post-initialization processing to generate the concatenated name and set it as the `name` attribute.

        Calls the parent class's __post_init__ method and creates a concatenated name
        by joining the names of all operator sets in `operators_set`.
        """
        # Generate the concatenated name from the operator sets
        concatenated_name = "_".join([op.name.value if hasattr(op.name, "value") else op.name for op in self.operators_set])
        # Set the inherited name attribute using `object.__setattr__` since the dataclass is frozen
        object.__setattr__(self, "name", concatenated_name)


@dataclass(frozen=True)
class Fusing(TargetPlatformModelComponent):
    """
    Fusing defines a tuple of operators that should be combined and treated as a single operator,
    hence no quantization is applied between them.

    Attributes:
        operator_groups (Tuple[Union[OperatorsSet, OperatorSetConcat]]): A tuple of operator groups,
                                                                              each being either an OperatorSetConcat or an OperatorsSet.
        name (str): The name for the Fusing instance. If not provided, it is generated from the operator groups' names.
    """
    operator_groups: Tuple[Union[OperatorsSet, OperatorSetConcat]]

    def __post_init__(self):
        """
        Post-initialization processing for input validation and name generation.

        Calls the parent class's __post_init__ method, validates the operator_groups,
        and generates the name if not explicitly provided.

        Raises:
            Logger critical if operator_groups is not a tuple or if it contains fewer than two operators.
        """
        # Validate the operator_groups
        if not isinstance(self.operator_groups, tuple):
            Logger.critical(
                f"Operator groups should be of type 'tuple' but is {type(self.operator_groups)}.") # pragma: no cover
        if len(self.operator_groups) < 2:
            Logger.critical("Fusing cannot be created for a single operator.") # pragma: no cover

        # Generate the name from the operator groups if not provided
        generated_name = '_'.join([x.name.value if hasattr(x.name, 'value') else x.name for x in self.operator_groups])
        object.__setattr__(self, 'name', generated_name)

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
            return {self.name: ' -> '.join([x.name for x in self.operator_groups])}
        return ' -> '.join([x.name for x in self.operator_groups])


@dataclass(frozen=True)
class TargetPlatformModel:
    """
    Represents the hardware configuration used for quantized model inference.

    Attributes:
        default_qco (QuantizationConfigOptions): Default quantization configuration options for the model.
        tpc_minor_version (Optional[int]): Minor version of the Target Platform Configuration.
        tpc_patch_version (Optional[int]): Patch version of the Target Platform Configuration.
        tpc_platform_type (Optional[str]): Type of the platform for the Target Platform Configuration.
        add_metadata (bool): Flag to determine if metadata should be added.
        name (str): Name of the Target Platform Model.
        operator_set (Tuple[OperatorsSetBase]): Tuple of operator sets within the model.
        fusing_patterns (Tuple[Fusing]): Tuple of fusing patterns for the model.
        is_simd_padding (bool): Indicates if SIMD padding is applied.
        SCHEMA_VERSION (int): Version of the schema for the Target Platform Model.
    """
    default_qco: QuantizationConfigOptions
    tpc_minor_version: Optional[int]
    tpc_patch_version: Optional[int]
    tpc_platform_type: Optional[str]
    add_metadata: bool = True
    name: str = "default_tp_model"
    operator_set: Tuple[OperatorsSetBase] = None
    fusing_patterns: Tuple[Fusing] = None
    is_simd_padding: bool = False

    SCHEMA_VERSION: int = 1

    def __post_init__(self):
        """
        Post-initialization processing for input validation.

        Raises:
            Logger critical if the default_qco is not an instance of QuantizationConfigOptions
            or if it contains more than one quantization configuration.
        """
        # Validate `default_qco`
        if not isinstance(self.default_qco, QuantizationConfigOptions):
            Logger.critical("'default_qco' must be an instance of QuantizationConfigOptions.") # pragma: no cover
        if len(self.default_qco.quantization_configurations) != 1:
            Logger.critical("Default QuantizationConfigOptions must contain exactly one option.") # pragma: no cover

        opsets_names = [op.name.value if hasattr(op.name, "value") else op.name for op in self.operator_set] if self.operator_set else []
        if len(set(opsets_names)) != len(opsets_names):
            Logger.critical("Operator Sets must have unique names.")  # pragma: no cover

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