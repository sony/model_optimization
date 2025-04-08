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
from tests_pytest._fw_tests_common_base.base_mct_bypass_test import BaseMCTBypassTest


class BasePytorchMCTBypassTest(BaseMCTBypassTest):


    def _build_test_model(self):

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
                self.bn = nn.BatchNorm2d(8)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                return x

        return Model()

    def _assert_models_equal(self, model, out_model):
        sd_model = model.state_dict()
        sd_out_model = out_model.state_dict()
        assert sd_model.keys() == sd_out_model.keys(), "Input and output Models state_dict keys differ."

        for key in sd_model:
            assert torch.equal(sd_model[key], sd_out_model[key]), f"Mismatch in parameter '{key}' between input and output models."