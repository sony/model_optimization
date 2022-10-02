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
import torch
from torch.nn import Conv2d, Linear, BatchNorm2d, ConvTranspose2d, Hardtanh, ReLU, ReLU6
from torch.nn.functional import relu, relu6, hardtanh

from model_compression_toolkit.core.tpc_models.qnnpack_tpc.v1.tp_model import get_tp_model
import model_compression_toolkit as mct
from model_compression_toolkit.core.tpc_models.qnnpack_tpc.v1 import __version__ as TPC_VERSION

tp = mct.target_platform


def get_pytorch_tpc() -> tp.TargetPlatformCapabilities:
    """
    get a Pytorch TargetPlatformCapabilities object with default operation sets to layers mapping.
    Returns: a Pytorch TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    qnnpack_pytorch = get_tp_model()
    return generate_pytorch_tpc(name='qnnpack_pytorch', tp_model=qnnpack_pytorch)


def generate_pytorch_tpc(name: str, tp_model: tp.TargetPlatformModel):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.
    Args:
        name: Name of the TargetPlatformModel.
        tp_model: TargetPlatformModel object.
    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    pytorch_tpc = tp.TargetPlatformCapabilities(tp_model,
                                                name=name,
                                                version=TPC_VERSION)

    with pytorch_tpc:
        tp.OperationsSetToLayers("Conv", [Conv2d,
                                          torch.nn.functional.conv2d,
                                          ConvTranspose2d,
                                          torch.nn.functional.conv_transpose2d])

        tp.OperationsSetToLayers("Linear", [Linear])

        tp.OperationsSetToLayers("BatchNorm", [BatchNorm2d])

        tp.OperationsSetToLayers("Relu", [torch.relu,
                                          ReLU,
                                          ReLU6,
                                          relu,
                                          relu6,
                                          tp.LayerFilterParams(Hardtanh, min_val=0),
                                          tp.LayerFilterParams(hardtanh, min_val=0)])

    return pytorch_tpc
