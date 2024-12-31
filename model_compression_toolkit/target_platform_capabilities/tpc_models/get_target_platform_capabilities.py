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
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tp_model import get_tp_model


# TODO: These methods need to be replaced once modifying the TPC API.


def get_target_platform_capabilities(fw_name: str,
                                     target_platform_name: str,
                                     target_platform_version: str = None) -> TargetPlatformModel:
    """
    This is a degenerated function that only returns the MCT default TargetPlatformModel object, to comply with the
    existing TPC API.

    Args:
        fw_name: Framework name of the TargetPlatformCapabilities (not used in this function).
        target_platform_name: Target platform model name the model will use for inference (not used in this function).
        target_platform_version: Target platform capabilities version (not used in this function).

    Returns:
        A default TargetPlatformModel object.
    """

    assert fw_name == DEFAULT_TP_MODEL or fw_name == 'v1', \
        "The usage of get_target_platform_capabilities API is supported only with the default TPC ('v1')."
    return get_tp_model()


def get_tpc_model(name: str, tp_model: TargetPlatformModel):
    """
    This is a utility method that just returns the TargetPlatformModel that it receives, to support existing TPC API.

    Args:
        name: the name of the TargetPlatformModel (not used in this function).
        tp_model: a TargetPlatformModel to return.

    Returns:
        The given TargetPlatformModel object.

    """

    return tp_model
