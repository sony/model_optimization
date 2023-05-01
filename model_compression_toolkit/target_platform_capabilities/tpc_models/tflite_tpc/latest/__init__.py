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
from model_compression_toolkit.constants import FOUND_TF, FOUND_TORCH
from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tp_model import get_tp_model, generate_tp_model, get_op_quantization_configs
if FOUND_TF:
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc_keras import get_keras_tpc as get_keras_tpc_latest
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc_keras import generate_keras_tpc
if FOUND_TORCH:
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc_pytorch import get_pytorch_tpc as get_pytorch_tpc_latest
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc_pytorch import generate_pytorch_tpc