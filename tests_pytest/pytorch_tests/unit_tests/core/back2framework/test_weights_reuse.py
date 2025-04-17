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
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph, graph_preparation_runner
from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PytorchModel
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch

import torch
from torch import nn


def data_gen():
    yield [torch.rand(1, 3, 5, 5)]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 3 * 3, 10)
        self.fc2 = nn.Linear(8 * 3 * 3, 10)

    def forward(self, x):
        batch_size = x.size(0)

        x1 = self.shared_conv(x)  # shape: (batch_size, 8, height-2, width-2) = (1, 8, 3, 3)
        x2 = self.shared_conv(x)  # shape: (batch_size, 8, height-2, width-2) = (1, 8, 3, 3)
        x1 = x1.view(batch_size, -1)  # shape: (batch_size, 8 * 3 * 3)
        x2 = x2.view(batch_size, -1)  # shape: (batch_size, 8 * 3 * 3)
        x1 = self.fc1(x1)  # shape: (batch_size, 10)
        x2 = self.fc2(x2)  # shape: (batch_size, 10)
        x = x1 + x2
        return x


def get_model_graph(model, minimal_tpc):
    fw_impl = PytorchImplementation()
    fw_info = DEFAULT_PYTORCH_INFO
    return graph_preparation_runner(model,
                                    data_gen,
                                    QuantizationConfig(),
                                    fw_info=fw_info,
                                    fw_impl=fw_impl,
                                    fqc=AttachTpcToPytorch().attach(minimal_tpc),
                                    mixed_precision_enable=False,
                                    running_gptq=False)


def test_weights_reuse_toposort(minimal_tpc):
    """
    Test that reused nodes are successfully initiated after their group node was initiated.
    Test it with nodes sorted in topological order.
    """
    model = Model()
    graph = get_model_graph(model, minimal_tpc)
    pytorch_model = PytorchModel(graph=graph)
    assert len(pytorch_model._reused_nodes) == 1


def test_weights_reuse_reversed_toposort(minimal_tpc):
    """
    Test that reused nodes are successfully initiated after their group node was initiated.
    Test it with nodes sorted in reversed topological order.
    """
    model = Model()
    graph = get_model_graph(model, minimal_tpc)
    pytorch_model = PytorchModel(graph=graph)

    pytorch_model.node_sort.reverse()
    pytorch_model.reuse_groups = {}
    pytorch_model._reused_nodes = []
    pytorch_model._add_all_modules()
    assert len(pytorch_model._reused_nodes) == 1
