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
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest

"""
This tests checks the 'reshape_with_static_shapes' substitution.
We create a model with dynamic input shape attributes to the operators 'reshape' and 'view'.
We check that the model after conversion replaces the attributes with static-list-of-ints attributes.
"""
class ReshapeNet(torch.nn.Module):
    def __init__(self):
        super(ReshapeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        batch, channels, height, width = x.size()
        x = x * height
        channels = channels + width
        channels = channels - width
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, height, width, channels)
        batch, channels, height, width = x.size()
        height = height + batch
        height = height - batch
        x = torch.transpose(x, 1, 2)
        return x.reshape(-1, channels, height, width)


class ReshapeNetTest(BasePytorchTest):
    """
    This tests checks the 'reshape_with_static_shapes' substitution.
    We create a model with dynamic input shape attributes to the operators 'reshape' and 'view'.
    We check that the model after conversion replaces the attributes with static-list-of-ints attributes.
    """
    def __init__(self, unit_test):
        super().__init__(unit_test, convert_to_fx=False)

    def create_feature_network(self, input_shape):
        return ReshapeNet()

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):

        ######################################################
        # check the 'reshape_with_static_shapes' substitution:
        ######################################################

        for k, v in quantized_models.items():
            # check that only one 'torch.Tensor.size' node is still in the graph. There are initially two.
            # One should remain and one should be deleted in the substitution.
            assert (sum([n.type == torch.Tensor.size for n in v.graph.nodes]) == 1)

            # check that the reshape attributes are lists and not nodes in the graph
            reshape_nodes = [n for n in v.graph.nodes if (n.type == torch.reshape or n.type == torch.Tensor.view)]
            for r in reshape_nodes:
                for o in r.op_call_args:
                    assert isinstance(o, list)

        ######################################################
        # check the all other comparisons:
        ######################################################
        super(ReshapeNetTest, self).compare(quantized_models, float_model, input_x=input_x, quantization_info=quantization_info)