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


import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, ConvTranspose2d, Hardtanh, ReLU, ReLU6
from torch.nn.functional import relu, relu6, hardtanh

from model_compression_toolkit.common.target_platform.targetplatform2framework import \
    TargetPlatformCapabilities, LayerFilterParams
from model_compression_toolkit.common.target_platform.targetplatform2framework import \
    OperationsSetToLayers
from model_compression_toolkit.tpc_models.qnnpack import get_qnnpack_model


def get_qnnpack_pytorch():
    qnnpack_tp_model = get_qnnpack_model()
    qnnpack_pytorch = TargetPlatformCapabilities(qnnpack_tp_model,
                                                 name='qnnpack_pytorch')

    with qnnpack_pytorch:
        OperationsSetToLayers("Conv", [Conv2d,
                                       torch.nn.functional.conv2d,
                                       ConvTranspose2d,
                                       torch.nn.functional.conv_transpose2d])

        OperationsSetToLayers("Linear", [Linear])

        OperationsSetToLayers("BatchNorm", [BatchNorm2d])

        OperationsSetToLayers("Relu", [torch.relu,
                                       ReLU,
                                       ReLU6,
                                       relu,
                                       relu6,
                                       LayerFilterParams(Hardtanh, min_val=0),
                                       LayerFilterParams(hardtanh, min_val=0)])

    return qnnpack_pytorch
