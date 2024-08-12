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
from model_compression_toolkit.verify_packages import FOUND_TORCH, FOUND_TF
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.constants import LATEST

def get_tpc_dict_by_fw(fw_name):
    tpc_models_dict = None
    if fw_name == TENSORFLOW:
        ###############################
        # Build Tensorflow TPC models
        ###############################
        if FOUND_TF:
            from model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.v1.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v1
            from model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.latest import \
                get_keras_tpc_latest

            # Keras: TPC versioning
            tpc_models_dict = {'v1': get_keras_tpc_v1,
                               LATEST: get_keras_tpc_latest}
    elif fw_name == PYTORCH:
        ###############################
        # Build Pytorch TPC models
        ###############################
        if FOUND_TORCH:
            from model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.v1.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v1
            from model_compression_toolkit.target_platform_capabilities.tpc_models.qnnpack_tpc.latest import \
                get_pytorch_tpc_latest

            # Pytorch: TPC versioning
            tpc_models_dict = {'v1': get_pytorch_tpc_v1,
                               LATEST: get_pytorch_tpc_latest}
    if tpc_models_dict is not None:
        return tpc_models_dict
    else:
        Logger.critical(f'Framework {fw_name} is not supported in imx500 or the relevant packages are not '
                        f'installed. Please make sure the relevant packages are installed when using MCT for optimizing'
                        f' a {fw_name} model. For Tensorflow, please install tensorflow. For PyTorch, please install '
                        f'torch.')  # pragma: no cover
