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
from dataclasses import dataclass, asdict

from typing import Dict, Any
from model_compression_toolkit.constants import OPERATORS_SCHEDULING, FUSED_NODES_MAPPING, CUTS, MAX_CUT, OP_ORDER, \
    OP_RECORD, SHAPE, NODE_OUTPUT_INDEX, NODE_NAME, TOTAL_SIZE, MEM_ELEMENTS
from model_compression_toolkit.core.common.graph.memory_graph.compute_graph_max_cut import SchedulerInfo
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities import \
    FrameworkQuantizationCapabilities


def create_model_metadata(fqc: FrameworkQuantizationCapabilities,
                          scheduling_info: SchedulerInfo = None) -> Dict:
    """
    Creates and returns a metadata dictionary for the model, including version information
    and optional scheduling information.

    Args:
        fqc: A FQC object to get the version.
        scheduling_info: An object containing scheduling details and metadata. Default is None.

    Returns:
        Dict: A dictionary containing the model's version information and optional scheduling information.
    """
    _metadata = get_versions_dict(fqc)
    if scheduling_info:
        scheduler_metadata = get_scheduler_metadata(scheduler_info=scheduling_info)
        _metadata['scheduling_info'] = scheduler_metadata
    return _metadata


def get_versions_dict(fqc) -> Dict:
    """

    Returns: A dictionary with FQC, MCT and FQC-Schema versions.

    """
    # imported inside to avoid circular import error
    from model_compression_toolkit import __version__ as mct_version

    @dataclass
    class TPCVersions:
        mct_version: str
        tpc_minor_version: str = f'{fqc.tpc.tpc_minor_version}'
        tpc_patch_version: str = f'{fqc.tpc.tpc_patch_version}'
        tpc_platform_type: str = f'{fqc.tpc.tpc_platform_type}'
        tpc_name: str = f'{fqc.tpc.name}'
        tpc_schema: str = f'{fqc.tpc.SCHEMA_VERSION}'

    return asdict(TPCVersions(mct_version))


def get_scheduler_metadata(scheduler_info: SchedulerInfo) -> Dict[str, Any]:
    """
    Extracts and returns metadata from SchedulerInfo.

    Args:
        scheduler_info (SchedulerInfo): The scheduler information object containing scheduling details like cuts and
        fusing mapping.

    Returns:
        Dict[str, Any]: A dictionary containing extracted metadata, including schedule, maximum cut,
        cuts information, and fused nodes mapping.
    """
    scheduler_metadata = {
        OPERATORS_SCHEDULING: [str(layer) for layer in scheduler_info.operators_scheduling],
        MAX_CUT: scheduler_info.max_cut,
        CUTS: [
            {
                OP_ORDER: [op.name for op in cut.op_order],
                OP_RECORD: [op.name for op in cut.op_record],
                MEM_ELEMENTS: [
                    {
                        SHAPE: list(tensor.shape),
                        NODE_NAME: tensor.node_name,
                        TOTAL_SIZE: float(tensor.total_size),
                        NODE_OUTPUT_INDEX: tensor.node_output_index
                    }
                    for tensor in cut.mem_elements.elements
                ]
            }
            for cut in scheduler_info.cuts
        ],
        FUSED_NODES_MAPPING: scheduler_info.fused_nodes_mapping
    }
    return scheduler_metadata
