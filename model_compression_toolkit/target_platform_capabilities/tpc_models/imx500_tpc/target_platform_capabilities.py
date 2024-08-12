# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.logger import Logger

from model_compression_toolkit.constants import TENSORFLOW, PYTORCH
from model_compression_toolkit.verify_packages import FOUND_TORCH, FOUND_TF
from model_compression_toolkit.target_platform_capabilities.constants import LATEST


def get_tpc_dict_by_fw(fw_name):
    tpc_models_dict = None
    if fw_name == TENSORFLOW:
        ###############################
        # Build Tensorflow TPC models
        ###############################
        if FOUND_TF:
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
                get_keras_tpc_latest
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v1
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_lut.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v1_lut
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_pot.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v1_pot
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v2.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v2
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v2_lut.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v2_lut
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v3.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v3
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v3_lut.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v3_lut
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tpc_keras import \
                get_keras_tpc as get_keras_tpc_v4

            # Keras: TPC versioning
            tpc_models_dict = {'v1': get_keras_tpc_v1,
                               'v1_lut': get_keras_tpc_v1_lut,
                               'v1_pot': get_keras_tpc_v1_pot,
                               'v2': get_keras_tpc_v2,
                               'v2_lut': get_keras_tpc_v2_lut,
                               'v3': get_keras_tpc_v3,
                               'v3_lut': get_keras_tpc_v3_lut,
                               'v4': get_keras_tpc_v4,
                               LATEST: get_keras_tpc_latest}
    elif fw_name == PYTORCH:
        ###############################
        # Build Pytorch TPC models
        ###############################
        if FOUND_TORCH:
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
                get_pytorch_tpc_latest
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v1
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_pot.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v1_pot
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1_lut.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v1_lut
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v2.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v2
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v2_lut.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v2_lut
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v3.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v3
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v3_lut.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v3_lut
            from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v4.tpc_pytorch import \
                get_pytorch_tpc as get_pytorch_tpc_v4

            # Pytorch: TPC versioning
            tpc_models_dict = {'v1': get_pytorch_tpc_v1,
                               'v1_lut': get_pytorch_tpc_v1_lut,
                               'v1_pot': get_pytorch_tpc_v1_pot,
                               'v2': get_pytorch_tpc_v2,
                               'v2_lut': get_pytorch_tpc_v2_lut,
                               'v3': get_pytorch_tpc_v3,
                               'v3_lut': get_pytorch_tpc_v3_lut,
                               'v4': get_pytorch_tpc_v4,
                               LATEST: get_pytorch_tpc_latest}
    if tpc_models_dict is not None:
        return tpc_models_dict
    else:
        Logger.critical(f'Framework {fw_name} is not supported in imx500 or the relevant packages are not '
                        f'installed. Please make sure the relevant packages are installed when using MCT for optimizing'
                        f' a {fw_name} model. For Tensorflow, please install tensorflow. For PyTorch, please install '
                        f'torch.')  # pragma: no cover
