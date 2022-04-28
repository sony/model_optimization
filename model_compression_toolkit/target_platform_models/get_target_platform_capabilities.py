# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

import importlib

from model_compression_toolkit.common.constants import TENSORFLOW, PYTORCH
from model_compression_toolkit.common.target_platform import TargetPlatformCapabilities


#############################
# Build Tensorflow models:
#############################

found_tf = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("tensorflow_model_optimization") is not None
tf_models_dict = {}

if found_tf:
    from model_compression_toolkit.target_platform_models.keras_target_platforms.keras_default import get_default_tpc_keras
    from model_compression_toolkit.target_platform_models.keras_target_platforms.keras_tflite import get_keras_tflite_tpc
    from model_compression_toolkit.target_platform_models.keras_target_platforms.keras_qnnpack import get_qnnpack_tensorflow
    from model_compression_toolkit.keras.constants import DEFAULT_TP_MODEL, TFLITE_TP_MODEL, QNNPACK_TP_MODEL

    tf_models_dict = {DEFAULT_TP_MODEL: get_default_tpc_keras(),
                      TFLITE_TP_MODEL: get_keras_tflite_tpc(),
                      QNNPACK_TP_MODEL: get_qnnpack_tensorflow()}


#############################
# Build Pytorch models:
#############################
found_torch = importlib.util.find_spec("torch") is not None
torch_models_dict = {}

if found_torch:
    from model_compression_toolkit.target_platform_models.pytorch_target_platforms.pytorch_default import get_default_tpc_pytorch
    from model_compression_toolkit.target_platform_models.pytorch_target_platforms.pytorch_qnnpack import get_qnnpack_pytorch
    from model_compression_toolkit.target_platform_models.pytorch_target_platforms.pytorch_tflite import get_pytorch_tflite_model
    from model_compression_toolkit.pytorch.constants import DEFAULT_TP_MODEL, TFLITE_TP_MODEL, QNNPACK_TP_MODEL

    torch_models_dict = {DEFAULT_TP_MODEL: get_default_tpc_pytorch(),
                         TFLITE_TP_MODEL: get_pytorch_tflite_model(),
                         QNNPACK_TP_MODEL: get_qnnpack_pytorch()}


tpc_dict = {TENSORFLOW: tf_models_dict,
            PYTORCH: torch_models_dict}


def get_target_platform_capabilities(fw_name: str,
                                     tp_model_name: str) -> TargetPlatformCapabilities:
    """
    Get a TargetPlatformCapabilities by the model name and the framework name.
    For now, it supports frameworks 'tensorflow' and 'pytorch'. For both of them
    the target platform model can be 'default','tflite', or 'qnnpack'.

    Args:
        fw_name (str): Framework name of the TargetPlatformCapabilities.
        tp_model_name (str): TargetPlatformModel name the model will use for inference.

    Returns:
        A TargetPlatformCapabilities object that models the hardware and attaches
        a framework information to it.
    """
    assert fw_name in tpc_dict, f'Framework {fw_name} is not supported'
    supported_models_by_fw = tpc_dict.get(fw_name)
    assert tp_model_name in supported_models_by_fw, f'TargetPlatformModel named {tp_model_name} is not supported for framework {fw_name}'
    return tpc_dict.get(fw_name).get(tp_model_name)
