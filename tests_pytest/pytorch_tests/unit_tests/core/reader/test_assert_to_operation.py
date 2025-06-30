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

import pytest
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import TorchFwMixin


def data_gen():
    yield [torch.rand(1, 10, 28, 32)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, kernel_size=(5, 4))
        self.conv2 = nn.Conv2d(20, 15, kernel_size=(4, 6), groups=5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x).to(x.device)
        return x


def test_assert_to_operation(minimal_tpc):
    Model()(next(data_gen())[0])

    model = Model()

    with pytest.raises(Exception, match=f'The call method "to" is not supported. Please consider moving "torch.Tensor.to" operations to init code.'):
        _ = read_model_to_graph(model,
                                data_gen,
                                fqc=TorchFwMixin.attach_to_fw_func(minimal_tpc),
                                fw_impl=TorchFwMixin.fw_impl)
