from typing import Any

import model_compression_toolkit.target_platform_capabilities.schema.v1 as schema_v1
import model_compression_toolkit.target_platform_capabilities.schema.v2 as schema_v2
import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema

ALL_SCHEMA_VERSIONS = [schema_v1]


def is_tpc_instance(tpc_obj_or_path: Any) -> bool:
    """"""
    return type(tpc_obj_or_path) in [
        schema_v1.TargetPlatformCapabilities,
        schema_v2.TargetPlatformCapabilities
    ]


def _schema_v1_to_v2(tpc: schema_v1.TargetPlatformCapabilities) -> schema.TargetPlatformCapabilities:
    """
    Converts given tpc of schema version 1 to schema version 2
    :return: TargetPlatformCapabilities instance of of schema version 2
    """
    raise NotImplementedError("Once schema v2 is implemented, add necessary adaptations to _schema_v1_to_v2 function")
    return schema.TargetPlatformCapabilities(default_qco=tpc.default_qco,
                                             operator_set=tpc.operator_set,
                                             fusing_patterns=tpc.fusing_patterns,
                                             tpc_minor_version=tpc.tpc_minor_version,
                                             tpc_patch_version=tpc.tpc_patch_version,
                                             tpc_platform_type=tpc.tpc_platform_type,
                                             add_metadata=tpc.add_metadata)


def tpc_to_current_schema_version(tpc: schema.TargetPlatformCapabilities):
    """
    Given tpc instance of some schema version, convert it to the current MCT schema version.

    In case a new schema is added to MCT, need to add a conversion function from the previous version to the new
    version, e.g. if the current schema version was updated from v4 to v5, need to add _schema_v4_to_v5 function to
    this file, than and add it to the conversion_map.

    :param tpc: TargetPlatformCapabilities of some schema version
    :return: TargetPlatformCapabilities with the current MCT schema version
    """
    conversion_map = {
        schema_v1.TargetPlatformCapabilities.SCHEMA_VERSION: _schema_v1_to_v2,
    }
    while tpc.SCHEMA_VERSION < schema.TargetPlatformCapabilities.SCHEMA_VERSION:
        if tpc.SCHEMA_VERSION not in conversion_map:
            raise Exception(f"TPC using schema version {tpc.SCHEMA_VERSION} which is not in schemas conversion map. "
                            f"Make sure the schema version is supported, or add it in case it's a new schema version")
        tpc = conversion_map[tpc.SCHEMA_VERSION](tpc)
    return tpc
