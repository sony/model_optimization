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

from model_compression_toolkit.core import QuantizationConfig
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest

"""
this checks that thresold prior to concat have been updated correctly.
"""

class ConcatNet(torch.nn.Module):
    '''
    create dummy concat network
    '''
    def __init__(self):
        super(ConcatNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=4)

    def forward(self, x):
        pre_concat_1 = self.conv1(x)
        pre_concat_2 = self.conv2(x)
        outputs = torch.cat((pre_concat_1, 8*pre_concat_2), dim=1)
        return outputs


class ConcatUpdateTest(BasePytorchFeatureNetworkTest):
    """
    This tests that all thresholds are equal
    """

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_quantization_config(self):
        return QuantizationConfig(concat_threshold_update=True)

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32], [self.val_batch_size, 3, 32, 32]]

    def create_networks(self):
        return ConcatNet()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        conv1_threshold = quantized_model.conv1_activation_holder_quantizer.activation_holder_quantizer.threshold_np
        multi_threshold = quantized_model.mul_activation_holder_quantizer.activation_holder_quantizer.threshold_np
        concat_threshold = quantized_model.cat_activation_holder_quantizer.activation_holder_quantizer.threshold_np

        self.unit_test.assertTrue(conv1_threshold == concat_threshold and multi_threshold == concat_threshold)

