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
import copy
from typing import Any, Union

import model_compression_toolkit.target_platform_capabilities.schema.v1 as schema_v1
import model_compression_toolkit.target_platform_capabilities.schema.v2 as schema_v2
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as current_schema

ALL_SCHEMA_VERSIONS = [schema_v1, schema_v2]  # needs to be updated with all active schema versions
FUTURE_SCHEMA_VERSIONS = []  # once future schema becomes current schema, move to it ALL_SCHEMA_VERSIONS
all_tpc_types = tuple([s.TargetPlatformCapabilities for s in ALL_SCHEMA_VERSIONS])
tpc_or_str_type = all_tpc_types + (str,)


def is_tpc_instance(tpc_obj_or_path: Any) -> bool:
    """
    Checks if the given object is an instance of a TargetPlatformCapabilities
    :param tpc_obj_or_path: Object to check its type
    :return: True if the given object is an instance of a TargetPlatformCapabilities, False otherwise
    """
    return type(tpc_obj_or_path) in all_tpc_types


def get_schema_by_version(schema_version: str):
    return {
        "1": schema_v1,
        "2": schema_v2
    }[schema_version]


def _schema_v1_to_v2(
        tpc: schema_v1.TargetPlatformCapabilities) -> schema_v2.TargetPlatformCapabilities:
    """
    Converts given tpc of schema version 1 to schema version 2.
    Schema v2 updates:
    1. New TPC field: insert_preserving_quantizers
        Compatability behavior: Set field to False by default
    2. New field in Fusing: fuse_op_quantization_config
        Compatability behavior: set enable_activation_quantization=False in every fuse_op_quantization_config
    3. New operator set names: EXP, SIN, COS
        Compatability behavior: Not required
    :return: TargetPlatformCapabilities instance of schema version 2
    """
    v1_default_qco = tpc.default_qco.base_config
    v2_default_qco = schema_v2.OpQuantizationConfig(
        default_weight_attr_config=v1_default_qco.default_weight_attr_config,
        attr_weights_configs_mapping=v1_default_qco.attr_weights_configs_mapping,
        activation_quantization_method=v1_default_qco.activation_quantization_method,
        activation_n_bits=v1_default_qco.activation_n_bits,
        supported_input_activation_n_bits=v1_default_qco.supported_input_activation_n_bits,
        enable_activation_quantization=False,  # set to False by default because feature not exist in schema v1
        quantization_preserving=v1_default_qco.quantization_preserving,
        fixed_scale=v1_default_qco.fixed_scale,
        fixed_zero_point=v1_default_qco.fixed_zero_point,
        simd_size=v1_default_qco.simd_size,
        signedness=v1_default_qco.signedness)

    schema_v2_fusing_patters = []
    for fussing_pattern in tpc.fusing_patterns:
        schema_v2_fusing_patters.append(
            schema_v2.Fusing(operator_groups=fussing_pattern.operator_groups,
                             fuse_op_quantization_config=copy.deepcopy(v2_default_qco),
                             name=fussing_pattern.name))

    tpc_schema_v2 = schema_v2.TargetPlatformCapabilities(default_qco=tpc.default_qco,
                                                         operator_set=tpc.operator_set,
                                                         fusing_patterns=schema_v2_fusing_patters,
                                                         tpc_minor_version=tpc.tpc_minor_version,
                                                         tpc_patch_version=tpc.tpc_patch_version,
                                                         tpc_platform_type=tpc.tpc_platform_type,
                                                         add_metadata=tpc.add_metadata,
                                                         insert_preserving_quantizers=False)  # set to False by default because feature not exist in schema v1
    return tpc_schema_v2


def get_conversion_map() -> dict:
    """
    Retrieves the schema conversion map.
    :return: A dictionary where:
        - Keys representing supported source schema versions.
        - Values are Callable functions that take tpc in one schema version and return it in the next (higher) version
    """
    conversion_map = {
        schema_v1.TargetPlatformCapabilities: _schema_v1_to_v2,
    }
    return conversion_map


def tpc_to_current_schema_version(
        tpc: Union[all_tpc_types]) -> current_schema.TargetPlatformCapabilities:
    """
    Given tpc instance of some schema version, convert it to the current MCT schema version.

    In case a new schema is added to MCT, need to add a conversion function from the previous version to the new
    version, e.g. if the current schema version was updated from v4 to v5, need to add _schema_v4_to_v5 function to
    this file, and add it to the conversion_map.

    :param tpc: TargetPlatformCapabilities of some schema version
    :return: TargetPlatformCapabilities with the current MCT schema version
    """
    conversion_map = get_conversion_map()
    prev_tpc_type = type(tpc)
    while not isinstance(tpc, current_schema.TargetPlatformCapabilities):
        if type(tpc) not in conversion_map:
            raise KeyError(f"TPC using schema version {tpc.SCHEMA_VERSION} which is not in schemas conversion map. "
                           f"Make sure the schema version is supported, or add it in case it's a new schema version")
        tpc = conversion_map[type(tpc)](tpc)
        if isinstance(tpc, prev_tpc_type):
            raise RuntimeError(f"TPC of type {prev_tpc_type} failed to update to next schema version")
        prev_tpc_type = type(tpc)
    return tpc
