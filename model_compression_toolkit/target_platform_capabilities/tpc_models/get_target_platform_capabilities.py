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
from model_compression_toolkit.constants import TENSORFLOW, PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL, IMX500_TP_MODEL, \
    TFLITE_TP_MODEL, QNNPACK_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tpc import get_tpc as get_tpc_imx500_v1
from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc import get_tpc as get_tpc_tflite_v1
from model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.v1.tpc import get_tpc as get_tpc_qnnpack_v1


# TODO: These methods need to be replaced once modifying the TPC API.

def get_target_platform_capabilities(fw_name: str,
                                     target_platform_name: str,
                                     target_platform_version: str = None) -> TargetPlatformCapabilities:
    """
    This is a degenerated function that only returns the MCT default TargetPlatformCapabilities object, to comply with the
    existing TPC API.

    Args:
        fw_name: Framework name of the FrameworkQuantizationCapabilities.
        target_platform_name: Target platform model name the model will use for inference.
        target_platform_version: Target platform capabilities version.

    Returns:
        A default TargetPlatformCapabilities object.
    """

    assert fw_name in [TENSORFLOW, PYTORCH], f"Unsupported framework {fw_name}."

    if target_platform_name == DEFAULT_TP_MODEL:
        return get_tpc_imx500_v1()

    assert target_platform_version == 'v1' or target_platform_version is None, \
        "The usage of get_target_platform_capabilities API is supported only with the default TPC ('v1')."

    if target_platform_name == IMX500_TP_MODEL:
        return get_tpc_imx500_v1()
    elif target_platform_name == TFLITE_TP_MODEL:
        return get_tpc_tflite_v1()
    elif target_platform_name == QNNPACK_TP_MODEL:
        return get_tpc_qnnpack_v1()

    raise ValueError(f"Unsupported target platform name {target_platform_name}.")


def get_tpc_model(name: str, tpc: TargetPlatformCapabilities):
    """
    This is a utility method that just returns the TargetPlatformCapabilities that it receives, to support existing TPC API.

    Args:
        name: the name of the TargetPlatformCapabilities (not used in this function).
        tpc: a TargetPlatformCapabilities to return.

    Returns:
        The given TargetPlatformCapabilities object.

    """

    return tpc
