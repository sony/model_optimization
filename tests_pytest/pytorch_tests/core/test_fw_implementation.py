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
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.graph_prep_runner import graph_preparation_runner
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation


def data_gen():
    yield [torch.rand(1, 10, 28, 32)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 20, kernel_size=(5, 4))
        self.conv2 = nn.Conv2d(20, 15, kernel_size=(4, 6), groups=5)
        self.conv3 = nn.Conv2d(15, 8, kernel_size=(3, 3), stride=2)
        self.conv4 = nn.Conv2d(8, 12, kernel_size=(3, 3), dilation=2)
        self.convtr1 = nn.ConvTranspose2d(12, 20, kernel_size=(5, 3))
        self.convtr2 = nn.ConvTranspose2d(20, 10, kernel_size=(3, 3), stride=2)
        self.convtr3 = nn.ConvTranspose2d(10, 5, kernel_size=(3, 3), dilation=2)
        self.dwconv1 = nn.Conv2d(5, 20, kernel_size=(2, 3), groups=5)
        self.dwconv2 = nn.Conv2d(20, 40, kernel_size=(3, 3), groups=20, stride=3)
        self.dwconv3 = nn.Conv2d(40, 80, kernel_size=(3, 3), groups=40, dilation=2)
        self.fc1 = nn.Linear(80, 10)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(120, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.convtr1(x)
        x = self.convtr2(x)
        x = self.convtr3(x)
        x = self.dwconv1(x)
        x = self.dwconv2(x)
        x = self.dwconv3(x)
        x = torch.permute(x, [0, 2, 3, 1])
        x = self.fc1(x)
        x = self.flatten(x)
        x = self.fc2(x)
        return x


def test_get_mac(minimal_tpc):
    Model()(next(data_gen())[0])

    fw_impl = PytorchImplementation()
    fw_info = DEFAULT_PYTORCH_INFO
    model = Model()

    graph = graph_preparation_runner(model,
                                     data_gen,
                                     QuantizationConfig(linear_collapsing=False),
                                     fw_info=fw_info,
                                     fw_impl=fw_impl,
                                     fqc=AttachTpcToPytorch().attach(minimal_tpc),
                                     mixed_precision_enable=False,
                                     running_gptq=False)

    nodes = graph.get_topo_sorted_nodes()
    # assert len(nodes) == 14, nodes
    assert fw_impl.get_node_mac_operations(nodes[0], fw_info) == 0
    assert fw_impl.get_node_mac_operations(nodes[1], fw_info) == (10*20*5*4)*24*29
    assert fw_impl.get_node_mac_operations(nodes[2], fw_info) == (4*3*4*6)*5*21*24
    assert fw_impl.get_node_mac_operations(nodes[3], fw_info) == (15*8*3*3)*10*11
    assert fw_impl.get_node_mac_operations(nodes[4], fw_info) == (8*12*3*3)*6*7
    assert fw_impl.get_node_mac_operations(nodes[5], fw_info) == (12*20*5*3)*10*9
    assert fw_impl.get_node_mac_operations(nodes[6], fw_info) == (20*10*3*3)*21*19
    assert fw_impl.get_node_mac_operations(nodes[7], fw_info) == (10*5*3*3)*25*23
    assert fw_impl.get_node_mac_operations(nodes[8], fw_info) == (5*2*3*4)*24*21
    assert fw_impl.get_node_mac_operations(nodes[9], fw_info) == (10*3*3*4)*8*7
    assert fw_impl.get_node_mac_operations(nodes[10], fw_info) == (40*3*3*2)*4*3
    assert fw_impl.get_node_mac_operations(nodes[10], fw_info) == (40*3*3*2)*4*3
    assert fw_impl.get_node_mac_operations(nodes[11], fw_info) == 0
    assert fw_impl.get_node_mac_operations(nodes[12], fw_info) == 4*3*(80*10)
    assert fw_impl.get_node_mac_operations(nodes[13], fw_info) == 0
    assert fw_impl.get_node_mac_operations(nodes[14], fw_info) == (4*3*10)*5


