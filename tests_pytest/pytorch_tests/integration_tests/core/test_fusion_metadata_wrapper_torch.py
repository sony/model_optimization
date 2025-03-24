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


from tests_pytest._fw_tests_common_base.base_fusion_metadata_wrapper_test import BaseGraphWithFusingMetadataTest
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin

import torch.nn as nn

class TestGraphWithFusionMetadataPytorch(BaseGraphWithFusingMetadataTest, TorchFwMixin):

    def _data_gen(self):
        return self.get_basic_data_gen(shapes=[(1, 3, 5, 5)])()

    def _get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, kernel_size=(3, 3))
                self.relu = nn.ReLU()
                self.flatten = nn.Flatten()
                self.linear = nn.Linear(in_features=27, out_features=10)
                self.softmax = nn.Softmax()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = self.flatten(x)
                x = self.linear(x)
                x = self.softmax(x)
                return x

        return Model()


