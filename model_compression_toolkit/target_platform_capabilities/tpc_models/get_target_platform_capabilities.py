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

from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.target_platform_capabilities import \
    get_tpc_dict_by_fw as get_imx500_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.target_platform_capabilities import \
    get_tpc_dict_by_fw as get_tflite_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.target_platform_capabilities import \
    get_tpc_dict_by_fw as get_qnnpack_tpc
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL, IMX500_TP_MODEL, TFLITE_TP_MODEL, QNNPACK_TP_MODEL,  LATEST

tpc_dict = {DEFAULT_TP_MODEL: get_imx500_tpc,
            IMX500_TP_MODEL: get_imx500_tpc,
            TFLITE_TP_MODEL: get_tflite_tpc,
            QNNPACK_TP_MODEL: get_qnnpack_tpc}


def get_target_platform_capabilities(fw_name: str,
                                     target_platform_name: str,
                                     target_platform_version: str = None) -> TargetPlatformCapabilities:
    """
    Get a TargetPlatformCapabilities by the target platform model name and the framework name.
    For now, it supports frameworks 'tensorflow' and 'pytorch'. For both of them
    the target platform model can be 'default', 'imx500', 'tflite', or 'qnnpack'.

    Args:
        fw_name: Framework name of the TargetPlatformCapabilities.
        target_platform_name: Target platform model name the model will use for inference.
        target_platform_version: Target platform capabilities version.
    Returns:
        A TargetPlatformCapabilities object that models the hardware and attaches
        a framework information to it.
    """
    assert target_platform_name in tpc_dict, f'Target platform {target_platform_name} is not defined!'
    fw_tpc = tpc_dict.get(target_platform_name)
    tpc_versions = fw_tpc(fw_name)
    if target_platform_version is None:
        target_platform_version = LATEST
    else:
        assert target_platform_version in tpc_versions, (f'TPC version {target_platform_version} is not supported for '
                                                         f'framework {fw_name}.')
    return tpc_versions[target_platform_version]()
