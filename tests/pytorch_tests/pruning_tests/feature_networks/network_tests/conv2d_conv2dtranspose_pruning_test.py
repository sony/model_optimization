# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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

import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import torch_tensor_to_numpy
import numpy as np

from tests.common_tests.pruninig.constant_importance_metric import add_const_importance_metric, ConstImportanceMetric
from tests.pytorch_tests.pruning_tests.feature_networks.pruning_pytorch_feature_test import PruningPytorchFeatureTest
import torch

from tests.pytorch_tests.utils import get_layers_from_model_by_type


class Conv2dConv2dTransposeModelForPrunning(torch.nn.Module):

    def __init__(self, use_bn, activation=None):
        super(Conv2dConv2dTransposeModelForPrunning, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(6)
        self.conv2dtrans2 = torch.nn.ConvTranspose2d(in_channels=6, out_channels=4, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1)
        self.activation = activation
        self.use_bn = use_bn

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        x = self.conv2dtrans2(x)
        x = self.conv3(x)
        return x

class Conv2dConv2dTransposePruningTest(PruningPytorchFeatureTest):
    """
    Test a network with two adjacent conv2d and check it's pruned a single group of channels.
    """

    def __init__(self,
                 unit_test,
                 use_bn=False,
                 activation_layer=None,
                 simd=1,
                 use_constant_importance_metric=True):

        super().__init__(unit_test,
                         input_shape=(3, 8, 8))
        self.use_bn = use_bn
        self.activation_layer = activation_layer
        self.simd = simd
        self.use_constant_importance_metric = use_constant_importance_metric

    def get_pruning_config(self):
        if self.use_constant_importance_metric:
            add_const_importance_metric(first_num_oc=6, second_num_oc=4, simd=self.simd)
            return mct.pruning.PruningConfig(importance_metric=ConstImportanceMetric.CONST)
        return super().get_pruning_config()

    def create_networks(self):
        return Conv2dConv2dTransposeModelForPrunning(use_bn=self.use_bn, activation=self.activation_layer)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        dense_convtrans_layers = get_layers_from_model_by_type(float_model, torch.nn.ConvTranspose2d)
        dense_conv_layers = get_layers_from_model_by_type(float_model, torch.nn.Conv2d)

        prunable_convtrans_layers = get_layers_from_model_by_type(quantized_model, torch.nn.ConvTranspose2d)
        prunable_conv_layers = get_layers_from_model_by_type(quantized_model, torch.nn.Conv2d)

        is_first_layer_pruned = prunable_conv_layers[0].out_channels == dense_conv_layers[
            0].out_channels - self.simd
        is_second_layer_pruned = prunable_convtrans_layers[0].out_channels == dense_convtrans_layers[0].out_channels - self.simd

        # Make sure only one of layers has been pruned
        self.unit_test.assertTrue(is_first_layer_pruned != is_second_layer_pruned)

        # In constant case, the last SIMD channels of the first layer should be pruned:
        if self.use_constant_importance_metric:
            self.unit_test.assertTrue(is_first_layer_pruned)
            self.unit_test.assertTrue(np.all(
                torch_tensor_to_numpy(prunable_conv_layers[0].weight) == torch_tensor_to_numpy(
                    dense_conv_layers[0].weight)[:-self.simd, :, :, :]))
            self.unit_test.assertTrue(np.all(
                torch_tensor_to_numpy(prunable_conv_layers[0].bias) == torch_tensor_to_numpy(
                    dense_conv_layers[0].bias)[:-self.simd]))

            # Make sure the only in channel removed is the last channel of the second conv layer
            self.unit_test.assertTrue(np.all(
                torch_tensor_to_numpy(prunable_convtrans_layers[0].weight) == torch_tensor_to_numpy(
                    dense_convtrans_layers[0].weight)[:-self.simd, :, :, :]))
            self.unit_test.assertTrue(np.all(
                torch_tensor_to_numpy(prunable_convtrans_layers[0].bias) == torch_tensor_to_numpy(
                    dense_convtrans_layers[0].bias)))

        if is_first_layer_pruned:
            self.unit_test.assertTrue(np.all(
                torch_tensor_to_numpy(prunable_conv_layers[1].weight) == torch_tensor_to_numpy(
                    dense_conv_layers[1].weight)))
            self.unit_test.assertTrue(np.all(
                torch_tensor_to_numpy(prunable_conv_layers[1].bias) == torch_tensor_to_numpy(
                    dense_conv_layers[1].bias)))

