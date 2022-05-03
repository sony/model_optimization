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

from model_compression_toolkit.common.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.common.target_platform.targetplatform2framework import OperationsSetToLayers, \
    LayerFilterParams
from model_compression_toolkit.common.target_platform.targetplatform2framework.attribute_filter import Eq
from model_compression_toolkit.tpc_models.tflite import get_tflite_tp_model
import operator


def get_pytorch_tflite_model():
    tflite_tp_model = get_tflite_tp_model()
    tflite_torch = TargetPlatformCapabilities(tflite_tp_model, name='tflite_torch')

    with tflite_torch:
        OperationsSetToLayers("PreserveQuantizationParams", [AvgPool2d,
                                                             avg_pool2d,
                                                             torch.cat,
                                                             torch.concat,
                                                             MaxPool2d,
                                                             max_pool2d,
                                                             torch.mul,
                                                             torch.multiply,
                                                             torch.reshape,
                                                             LayerFilterParams(interpolate, mode='bilinear'),
                                                             torch.nn.ZeroPad2d,
                                                             torch.gather,
                                                             torch.transpose,
                                                             torch.maximum,
                                                             torch.max,
                                                             torch.minimum,
                                                             torch.min,
                                                             torch.nn.functional.pad,
                                                             torch.select])

        OperationsSetToLayers("FullyConnected", [torch.nn.Linear,
                                                 torch.nn.functional.linear])

        OperationsSetToLayers("L2Normalization", [LayerFilterParams(torch.nn.functional.normalize,
                                                                    Eq('p',2) | Eq('p',None))])

        OperationsSetToLayers("LogSoftmax", [torch.nn.LogSoftmax])

        OperationsSetToLayers("Tanh", [torch.nn.Tanh,
                                       torch.nn.functional.tanh])

        OperationsSetToLayers("Softmax", [torch.nn.Softmax,
                                          torch.nn.functional.softmax])

        OperationsSetToLayers("Logistic", [torch.nn.Sigmoid,
                                           torch.nn.functional.sigmoid])

        OperationsSetToLayers("Conv2d", [torch.nn.Conv2d,
                                         torch.nn.functional.conv2d])

        OperationsSetToLayers("Relu", [torch.relu,
                                       torch.nn.ReLU,
                                       torch.nn.ReLU6,
                                       torch.nn.functional.relu,
                                       torch.nn.functional.relu6,
                                       LayerFilterParams(torch.nn.Hardtanh, min_val=0, max_val=6),
                                       LayerFilterParams(torch.nn.functional.hardtanh, min_val=0, max_val=6)])

        OperationsSetToLayers("Elu", [torch.nn.ELU,
                                      torch.nn.functional.elu])

        OperationsSetToLayers("BatchNorm", [torch.nn.BatchNorm2d,
                                            torch.nn.functional.batch_norm])

        OperationsSetToLayers("Squeeze", [torch.squeeze])

        OperationsSetToLayers("Add", [operator.add,
                                      torch.add])

    return tflite_torch


