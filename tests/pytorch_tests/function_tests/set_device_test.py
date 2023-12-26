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
import torch
from model_compression_toolkit.core.pytorch.pytorch_device_config import set_working_device, get_working_device
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

class SetDeviceTest(BasePytorchTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        set_working_device('cpu')
        device = get_working_device()
        self.unit_test.assertTrue(device == torch.device('cpu'))

        set_working_device('cuda')
        device = get_working_device()
        print(f'Device: {device}')
        self.unit_test.assertTrue(device in [torch.device('cuda'), torch.device('cuda:0')])

        set_working_device('cuda:0')
        device = get_working_device()
        print(f'Device: {device}')
        self.unit_test.assertTrue(device == torch.device('cuda:0'))

        set_working_device('cuda:999')
        device = get_working_device()
        print(f'Device: {device}')
        self.unit_test.assertFalse(device == torch.device('cuda:999'))