# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from torch import nn

from model_compression_toolkit.core import pytorch_resource_utilization_data
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from tests_pytest._fw_tests_common_base.base_ru_data_facade_test import BaseRUDataFacadeTest


class TestTorchRUDataFacade(BaseRUDataFacadeTest):
    fw_ru_data_facade = pytorch_resource_utilization_data

    def _build_model(self, input_shape, out_chan, kernel, const):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels=input_shape[1], out_channels=out_chan, kernel_size=kernel)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = torch.add(x, to_torch_tensor(const))
                return x

        return Model()
