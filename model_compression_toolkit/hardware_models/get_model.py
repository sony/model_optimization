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
from model_compression_toolkit.common.hardware_representation import FrameworkHardwareModel


#############################
# Build Tensorflow models:
#############################

found_tf = importlib.util.find_spec("tensorflow") is not None and importlib.util.find_spec("tensorflow_model_optimization") is not None
tf_models_dict = {}

if found_tf:
    from model_compression_toolkit.hardware_models.keras_hardware_model.keras_default import get_default_hwm_keras
    from model_compression_toolkit.hardware_models.keras_hardware_model.keras_tflite import get_keras_hardware_model_tflite
    from model_compression_toolkit.hardware_models.keras_hardware_model.keras_qnnpack import get_qnnpack_tensorflow
    from model_compression_toolkit.keras.constants import DEFAULT_HWM, TFLITE_HWM, QNNPACK_HWM

    tf_models_dict = {DEFAULT_HWM: get_default_hwm_keras(),
                      TFLITE_HWM: get_keras_hardware_model_tflite(),
                      QNNPACK_HWM: get_qnnpack_tensorflow()}


#############################
# Build Pytorch models:
#############################
found_torch = importlib.util.find_spec("torch") is not None
torch_models_dict = {}

if found_torch:
    from model_compression_toolkit.hardware_models.pytorch_hardware_model.pytorch_default import get_default_hwm_pytorch
    from model_compression_toolkit.hardware_models.pytorch_hardware_model.pytorch_qnnpack import get_qnnpack_pytorch
    from model_compression_toolkit.hardware_models.pytorch_hardware_model.pytorch_tflite import get_pytorch_tflite_model
    from model_compression_toolkit.pytorch.constants import DEFAULT_HWM, TFLITE_HWM, QNNPACK_HWM

    torch_models_dict = {DEFAULT_HWM: get_default_hwm_pytorch(),
                         TFLITE_HWM: get_pytorch_tflite_model(),
                         QNNPACK_HWM: get_qnnpack_pytorch()}


fw_hw_models_dict = {TENSORFLOW: tf_models_dict,
                     PYTORCH: torch_models_dict}


def get_model(fw_name: str,
              hw_name: str) -> FrameworkHardwareModel:
    """
    Get a FrameworkHardwareModel by the hardware model name and the framework name.
    For now, it supports frameworks 'tensorflow' and 'pytorch'. For both of them
    the hardware model can be 'default','tflite', or 'qnnpack'.

    Args:
        fw_name: Framework name of the FrameworkHardwareModel.
        hw_name: Hardware model name the model will use for inference.

    Returns:
        A FrameworkHardwareModel object that models the hardware and attaches
        a framework information to it.
    """
    assert fw_name in fw_hw_models_dict, f'Framework {fw_name} is not supported'
    supported_models_by_fw = fw_hw_models_dict.get(fw_name)
    assert hw_name in supported_models_by_fw, f'Hardware model named {hw_name} is not' \
                                              f' supported for framework {fw_name}'
    return fw_hw_models_dict.get(fw_name).get(hw_name)
