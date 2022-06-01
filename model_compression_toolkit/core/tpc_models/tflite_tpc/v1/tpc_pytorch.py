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
from torch.nn import AvgPool2d, MaxPool2d
from torch.nn.functional import avg_pool2d, max_pool2d, interpolate
from model_compression_toolkit.core.common.target_platform.targetplatform2framework.attribute_filter import Eq

from model_compression_toolkit.core.tpc_models.tflite_tpc.v1.tp_model import get_tp_model
import model_compression_toolkit as mct

tp = mct.target_platform


def get_pytorch_tpc() -> tp.TargetPlatformCapabilities:
    """
    get a Pytorch TargetPlatformCapabilities object with default operation sets to layers mapping.
    Returns: a Pytorch TargetPlatformCapabilities object for the given TargetPlatformModel.
    """
    tflite_tp_model = get_tp_model()
    return generate_pytorch_tpc(name='tflite_torch', tp_model=tflite_tp_model)


def generate_pytorch_tpc(name: str, tp_model: tp.TargetPlatformModel):
    """
    Generates a TargetPlatformCapabilities object with default operation sets to layers mapping.
    Args:
        name: Name of the TargetPlatformModel.
        tp_model: TargetPlatformModel object.
    Returns: a TargetPlatformCapabilities object for the given TargetPlatformModel.
    """

    pytorch_tpc = tp.TargetPlatformCapabilities(tp_model,
                                                name=name)

    with pytorch_tpc:
        tp.OperationsSetToLayers("NoQuantization", [AvgPool2d,
                                                    avg_pool2d,
                                                    torch.cat,
                                                    torch.concat,
                                                    MaxPool2d,
                                                    max_pool2d,
                                                    torch.mul,
                                                    torch.multiply,
                                                    torch.reshape,
                                                    tp.LayerFilterParams(interpolate, mode='bilinear'),
                                                    torch.nn.ZeroPad2d,
                                                    torch.gather,
                                                    torch.transpose,
                                                    torch.maximum,
                                                    torch.max,
                                                    torch.minimum,
                                                    torch.min,
                                                    torch.nn.functional.pad,
                                                    torch.select])

        tp.OperationsSetToLayers("FullyConnected", [torch.nn.Linear, torch.nn.functional.linear])
        tp.OperationsSetToLayers("L2Normalization",
                                 [tp.LayerFilterParams(torch.nn.functional.normalize, Eq('p', 2) | Eq('p', None))])
        tp.OperationsSetToLayers("LogSoftmax", [torch.nn.LogSoftmax])
        tp.OperationsSetToLayers("Tanh", [torch.nn.Tanh, torch.nn.functional.tanh])
        tp.OperationsSetToLayers("Softmax", [torch.nn.Softmax, torch.nn.functional.softmax])
        tp.OperationsSetToLayers("Logistic", [torch.nn.Sigmoid, torch.nn.functional.sigmoid])
        tp.OperationsSetToLayers("Conv2d", [torch.nn.Conv2d, torch.nn.functional.conv2d])
        tp.OperationsSetToLayers("Relu", [torch.relu,
                                          torch.nn.ReLU,
                                          torch.nn.ReLU6,
                                          torch.nn.functional.relu,
                                          torch.nn.functional.relu6,
                                          tp.LayerFilterParams(torch.nn.Hardtanh, min_val=0, max_val=6),
                                          tp.LayerFilterParams(torch.nn.functional.hardtanh, min_val=0, max_val=6)])
        tp.OperationsSetToLayers("Elu", [torch.nn.ELU, torch.nn.functional.elu])
        tp.OperationsSetToLayers("BatchNorm", [torch.nn.BatchNorm2d, torch.nn.functional.batch_norm])
        tp.OperationsSetToLayers("Squeeze", [torch.squeeze])

    return pytorch_tpc
