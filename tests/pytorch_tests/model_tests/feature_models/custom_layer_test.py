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
import random

import numpy as np
import torch

import model_compression_toolkit as mct
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_pytorch_tpc_latest
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This tests checks that custom layers are passing the quantization procedure.
"""


class ConvCustomLayer(torch.nn.Module):
    """
    Custom layer of Conv2D
    """

    def __init__(
            self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=2)

    def forward(self, x):
        B, C, H, W = x.shape
        if H == W:
            x = self.conv1(x)
        else:
            x = self.conv2(2)
        return x


class ConvBNCustomLayer(torch.nn.Module):
    """
    Custom layer of Conv2D and BatchNorm
    """

    def __init__(
            self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        if H == W:
            x = self.bn1(x)
        return x


class ConvCustomLayerNet(torch.nn.Module):
    def __init__(self):
        super(ConvCustomLayerNet, self).__init__()
        self.custom1 = ConvCustomLayer()
        self.bn1 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.custom1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        return x


class ConvBNCustomLayerNet(torch.nn.Module):
    def __init__(self):
        super(ConvBNCustomLayerNet, self).__init__()
        self.custom1 = ConvCustomLayer()
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.custom2 = ConvBNCustomLayer()

    def forward(self, x):
        x = self.custom1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.custom2(x)
        x = torch.relu(x)
        return x


class CustomLayerNetBaseTest(BasePytorchTest):
    """
    This base test checks that custom layers are passing the quantization procedure.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self, seed=0, experimental_facade=False, model_leafs=None):
        np.random.seed(seed)
        random.seed(a=seed)
        torch.random.manual_seed(seed)
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen_experimental():
            for _ in range(self.num_calibration_iter):
                yield x

        model_float = self.create_feature_network(input_shapes)
        core_config = self.get_core_config()
        tpc = get_pytorch_tpc_latest()
        for model_leaf in model_leafs:
            try:
                ptq_model, quantization_info = mct.pytorch_post_training_quantization_experimental(
                    in_module=model_float,
                    representative_data_gen=representative_data_gen_experimental,
                    target_kpi=self.get_kpi(),
                    core_config=core_config,
                    target_platform_capabilities=tpc,
                    new_experimental_exporter=self.experimental_exporter,
                    model_leaf_layers=model_leaf
                )
                self.compare(ptq_model, model_float, model_leaf=model_leaf)
            except Exception as e:
                # Should crash if the user didn't mention the Symbolic traced custom layers
                error_msg = e.message if hasattr(e, 'message') else str(e)
                self.unit_test.assertTrue(model_leaf is None, error_msg)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None, model_leaf=None):

        # check that the custom layer in the quantized model's layers
        for leaf_layer in model_leaf:
            self.unit_test.assertTrue(leaf_layer in [layer.__class__ for layer in quantized_model.modules()])


class ConvCustomLayerNetTest(CustomLayerNetBaseTest):
    """
    This tests checks that custom layer is passing the quantization procedure.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        return ConvCustomLayerNet()


class ConvBNCustomLayerNetTest(CustomLayerNetBaseTest):
    """
    This tests checks that 2 custom layers are passing the quantization procedure.
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_feature_network(self, input_shape):
        return ConvBNCustomLayerNet()
