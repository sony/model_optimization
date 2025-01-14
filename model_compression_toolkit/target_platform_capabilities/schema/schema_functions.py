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
from logging import Logger
from typing import Optional

from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OpQuantizationConfig, \
    TargetPlatformCapabilities, QuantizationConfigOptions, OperatorsSetBase


def max_input_activation_n_bits(op_quantization_config: OpQuantizationConfig) -> int:
    """
    Get the maximum supported input bit-width.

    Args:
        op_quantization_config (OpQuantizationConfig):  The configuration object from which to retrieve the maximum supported input bit-width.

    Returns:
        int: Maximum supported input bit-width.
    """
    return max(op_quantization_config.supported_input_activation_n_bits)


def get_config_options_by_operators_set(tpc: TargetPlatformCapabilities,
                                        operators_set_name: str) -> QuantizationConfigOptions:
    """
    Get the QuantizationConfigOptions of an OperatorsSet by its name.

    Args:
        tpc (TargetPlatformCapabilities): The target platform model containing the operator sets and their configurations.
        operators_set_name (str): The name of the OperatorsSet whose quantization configuration options are to be retrieved.

    Returns:
        QuantizationConfigOptions: The quantization configuration options associated with the specified OperatorsSet,
        or the default quantization configuration options if the OperatorsSet is not found.
    """
    for op_set in tpc.operator_set:
        if operators_set_name == op_set.name:
            return op_set.qc_options
    return tpc.default_qco


def get_default_op_quantization_config(tpc: TargetPlatformCapabilities) -> OpQuantizationConfig:
    """
    Get the default OpQuantizationConfig of the TargetPlatformCapabilities.

    Args:
        tpc (TargetPlatformCapabilities): The target platform model containing the default quantization configuration.

    Returns:
        OpQuantizationConfig: The default quantization configuration.

    Raises:
        AssertionError: If the default quantization configuration list contains more than one configuration option.
    """
    assert len(tpc.default_qco.quantization_configurations) == 1, \
        f"Default quantization configuration options must contain only one option, " \
        f"but found {len(tpc.default_qco.quantization_configurations)} configurations." # pragma: no cover
    return tpc.default_qco.quantization_configurations[0]


def is_opset_in_model(tpc: TargetPlatformCapabilities, opset_name: str) -> bool:
    """
    Check whether an OperatorsSet is defined in the model.

    Args:
        tpc (TargetPlatformCapabilities): The target platform model containing the list of operator sets.
        opset_name (str): The name of the OperatorsSet to check for existence.

    Returns:
        bool: True if an OperatorsSet with the given name exists in the target platform model,
              otherwise False.
    """
    return tpc.operator_set is not None and opset_name in [x.name for x in tpc.operator_set]

def get_opset_by_name(tpc: TargetPlatformCapabilities, opset_name: str) -> Optional[OperatorsSetBase]:
    """
    Get an OperatorsSet object from the model by its name.

    Args:
        tpc (TargetPlatformCapabilities): The target platform model containing the list of operator sets.
        opset_name (str): The name of the OperatorsSet to be retrieved.

    Returns:
        Optional[OperatorsSetBase]: The OperatorsSet object with the specified name if found.
        If no operator set with the specified name is found, None is returned.

    Raises:
        A critical log message if multiple operator sets with the same name are found.
    """
    opset_list = [x for x in tpc.operator_set if x.name == opset_name]
    if len(opset_list) > 1:
        Logger.critical(f"Found more than one OperatorsSet in TargetPlatformCapabilities with the name {opset_name}.") # pragma: no cover
    return opset_list[0] if opset_list else None
