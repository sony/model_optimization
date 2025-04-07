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
from typing import Dict, Any, Tuple, Optional

from pydantic import BaseModel, root_validator

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
    OperatorSetGroup,
    Fusing,
    OperatorSetNames)


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
        SCHEMA_VERSION (int): Version of the schema for the Target Platform Model.
    """
    default_qco: QuantizationConfigOptions
    operator_set: Optional[Tuple[OperatorsSet, ...]]
    fusing_patterns: Optional[Tuple[Fusing, ...]]
    tpc_minor_version: Optional[int]
    tpc_patch_version: Optional[int]
    tpc_platform_type: Optional[str]
    add_metadata: bool = True
    name: Optional[str] = "default_tpc"
    is_simd_padding: bool = False

    SCHEMA_VERSION: int = 2

    class Config:
        frozen = True

    @root_validator(allow_reuse=True)
    def validate_after_initialization(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform validation after the model has been instantiated.

        Args:
            values (Dict[str, Any]): The instantiated target platform model.

        Returns:
            Dict[str, Any]: The validated values.
        """
        # Validate `default_qco`
        default_qco = values.get('default_qco')
        if len(default_qco.quantization_configurations) != 1:
            Logger.critical("Default QuantizationConfigOptions must contain exactly one option.")  # pragma: no cover

        # Validate `operator_set` uniqueness
        operator_set = values.get('operator_set')
        if operator_set is not None:
            opsets_names = [
                op.name.value if isinstance(op.name, OperatorSetNames) else op.name
                for op in operator_set
            ]
            if len(set(opsets_names)) != len(opsets_names):
                Logger.critical("Operator Sets must have unique names.")  # pragma: no cover

        return values

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