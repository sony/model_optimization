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

from model_compression_toolkit.constants import FOUND_TF, FOUND_TORCH, TENSORFLOW, PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import LATEST


###############################
# Build Tensorflow TPC models
###############################
keras_tpc_models_dict = None
if FOUND_TF:
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc_keras import get_keras_tpc as get_keras_tpc_v1
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.latest import get_keras_tpc_latest

    # Keras: TPC versioning
    keras_tpc_models_dict = {'v1': get_keras_tpc_v1,
                             LATEST: get_keras_tpc_latest}

###############################
# Build Pytorch TPC models
###############################
pytorch_tpc_models_dict = None
if FOUND_TORCH:
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.v1.tpc_pytorch import \
        get_pytorch_tpc as get_pytorch_tpc_v1
    from model_compression_toolkit.target_platform_capabilities.tpc_models.tflite_tpc.latest import get_pytorch_tpc_latest

    # Pytorch: TPC versioning
    pytorch_tpc_models_dict = {'v1': get_pytorch_tpc_v1,
                               LATEST: get_pytorch_tpc_latest}

tpc_dict = {TENSORFLOW: keras_tpc_models_dict,
            PYTORCH: pytorch_tpc_models_dict}

