# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Dict, Any, Union, Tuple, Optional, Annotated

from pydantic import BaseModel, Field, root_validator, model_validator, ConfigDict

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.constants import FLOAT_BITWIDTH
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.schema.v1 import (
    Signedness,
    AttributeQuantizationConfig,
    OpQuantizationConfig,
    QuantizationConfigOptions,
    TargetPlatformModelComponent,
    OperatorsSetBase,
    OperatorsSet,
    OperatorSetGroup)


class OperatorSetNames(str, Enum):
    CONV = "Conv"
    DEPTHWISE_CONV = "DepthwiseConv2D"
    CONV_TRANSPOSE = "ConvTranspose"
    FULLY_CONNECTED = "FullyConnected"
    CONCATENATE = "Concatenate"
    STACK = "Stack"
    UNSTACK = "Unstack"
    GATHER = "Gather"
    EXPAND = "Expend"
    BATCH_NORM = "BatchNorm"
    L2NORM = "L2Norm"
    RELU = "ReLU"
    RELU6 = "ReLU6"
    LEAKY_RELU = "LeakyReLU"
    ELU = "Elu"
    HARD_TANH = "HardTanh"
    ADD = "Add"
    SUB = "Sub"
    MUL = "Mul"
    DIV = "Div"
    MIN = "Min"
    MAX = "Max"
    PRELU = "PReLU"
    ADD_BIAS = "AddBias"
    SWISH = "Swish"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    LOG_SOFTMAX = "LogSoftmax"
    TANH = "Tanh"
    GELU = "Gelu"
    HARDSIGMOID = "HardSigmoid"
    HARDSWISH = "HardSwish"
    FLATTEN = "Flatten"
    GET_ITEM = "GetItem"
    RESHAPE = "Reshape"
    UNSQUEEZE = "Unsqueeze"
    SQUEEZE = "Squeeze"
    PERMUTE = "Permute"
    TRANSPOSE = "Transpose"
    DROPOUT = "Dropout"
    SPLIT_CHUNK = "SplitChunk"
    MAXPOOL = "MaxPool"
    AVGPOOL = "AvgPool"
    SIZE = "Size"
    SHAPE = "Shape"
    EQUAL = "Equal"
    ARGMAX = "ArgMax"
    TOPK = "TopK"
    FAKE_QUANT = "FakeQuant"
    COMBINED_NON_MAX_SUPPRESSION = "CombinedNonMaxSuppression"
    ZERO_PADDING2D = "ZeroPadding2D"
    CAST = "Cast"
    RESIZE = "Resize"
    PAD = "Pad"
    FOLD = "Fold"
    STRIDED_SLICE = "StridedSlice"
    SSD_POST_PROCESS = "SSDPostProcess"
    EXP = "Exp"
    SIN = "Sin"
    COS = "Cos"

    @classmethod
    def get_values(cls):
        return [v.value for v in cls]


class Fusing(TargetPlatformModelComponent):
    """
    Fusing defines a tuple of operators that should be combined and treated as a single operator,
    hence no quantization is applied between them.

    Attributes:
        operator_groups (Tuple[Union[OperatorsSet, OperatorSetGroup], ...]): A tuple of operator groups,
                                                                              each being either an OperatorSetGroup or an OperatorsSet.
        fuse_op_quantization_config (Optional[OpQuantizationConfig]): The quantization configuration for the fused operator.
        name (Optional[str]): The name for the Fusing instance. If not provided, it is generated from the operator groups' names.
    """
    operator_groups: Tuple[Annotated[Union[OperatorsSet, OperatorSetGroup], Field(discriminator='type')], ...]
    fuse_op_quantization_config: Optional[OpQuantizationConfig] = None
    name: Optional[str] = None  # Will be set in the validator if not given.

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    def validate_and_set_name(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the operator_groups and set the name by concatenating operator group names.

        Args:
            values (Dict[str, Any]): Input data.

        Returns:
            Dict[str, Any]: Modified input data with 'name' set.
        """
        operator_groups = values.get('operator_groups')

        # When loading from JSON, lists are returned. If the value is a list, convert it to a tuple.
        if isinstance(operator_groups, list):
            values['operator_groups'] = tuple(operator_groups)

        if values.get('name') is None:
            # Generate the concatenated name from the operator groups
            concatenated_name = "_".join([
                op.name.value if isinstance(op.name, OperatorSetNames) else op.name
                for op in values['operator_groups']
            ])
            values['name'] = concatenated_name

        return values

    @model_validator(mode="after")
    def validate_after_initialization(cls, model: 'Fusing') -> Any:
        """
        Perform validation after the model has been instantiated.
        Ensures that there are at least two operator groups.
        """
        if len(model.operator_groups) < 2:
            Logger.critical("Fusing cannot be created for a single operator.")  # pragma: no cover
        
        return model

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
                        isinstance(self.operator_groups[i + j], OperatorSetGroup) and (
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


class TargetPlatformCapabilities(BaseModel):
    """
    Represents the hardware configuration used for quantized model inference.

    Attributes:
        default_qco (QuantizationConfigOptions): Default quantization configuration options for the model.
        operator_set (Optional[Tuple[OperatorsSet, ...]]): Tuple of operator sets within the model.
        fusing_patterns (Optional[Tuple[Fusing, ...]]): Tuple of fusing patterns for the model.
        tpc_minor_version (Optional[int]): Minor version of the Target Platform Configuration.
        tpc_patch_version (Optional[int]): Patch version of the Target Platform Configuration.
        tpc_platform_type (Optional[str]): Type of the platform for the Target Platform Configuration.
        add_metadata (bool): Flag to determine if metadata should be added.
        name (str): Name of the Target Platform Model.
        is_simd_padding (bool): Indicates if SIMD padding is applied.
        insert_preserving_quantizers (bool): Whether to include quantizers for quantization preserving operations in the quantized model.
        SCHEMA_VERSION (int): Version of the schema for the Target Platform Model.
    """
    default_qco: QuantizationConfigOptions
    operator_set: Optional[Tuple[OperatorsSet, ...]] = None
    fusing_patterns: Optional[Tuple[Fusing, ...]] = None
    tpc_minor_version: Optional[int] = None
    tpc_patch_version: Optional[int] = None
    tpc_platform_type: Optional[str] = None
    add_metadata: bool = True
    name: Optional[str] = "default_tpc"

    is_simd_padding: bool = False
    insert_preserving_quantizers: bool = False

    SCHEMA_VERSION: int = 2

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def validate_after_initialization(cls, model: 'TargetPlatformCapabilities') -> Any:
        """
        Perform validation after the model has been instantiated.

        Args:
            model (TargetPlatformCapabilities): The instantiated target platform model.

        Returns:
            TargetPlatformCapabilities: The validated model.
        """
        # Validate `default_qco`
        default_qco = model.default_qco
        if len(default_qco.quantization_configurations) != 1:
            Logger.critical("Default QuantizationConfigOptions must contain exactly one option.")  # pragma: no cover

        # Validate `operator_set` uniqueness
        operator_set = model.operator_set
        if operator_set is not None:
            opsets_names = [
                op.name.value if isinstance(op.name, OperatorSetNames) else op.name
                for op in operator_set
            ]
            if len(set(opsets_names)) != len(opsets_names):
                Logger.critical("Operator Sets must have unique names.")  # pragma: no cover

        return model

    def get_info(self) -> Dict[str, Any]:
        """
        Get a dictionary summarizing the TargetPlatformCapabilities properties.

        Returns:
            Dict[str, Any]: Summary of the TargetPlatformCapabilities properties.
        """
        return {
            "Model name": self.name,
            "Operators sets": [o.get_info() for o in self.operator_set] if self.operator_set else [],
            "Fusing patterns": [f.get_info() for f in self.fusing_patterns] if self.fusing_patterns else [],
        }

    def show(self):
        """
        Display the TargetPlatformCapabilities.
        """
        pprint.pprint(self.get_info(), sort_dicts=False)
