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
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
import numpy as np

from tests.common_tests.pruning.constant_importance_metric import add_const_importance_metric, ConstImportanceMetric
from tests.pytorch_tests.pruning_tests.feature_networks.pruning_pytorch_feature_test import PruningPytorchFeatureTest
import torch

from tests.pytorch_tests.utils import get_layers_from_model_by_type


class LinearModelForPrunning(torch.nn.Module):

    def __init__(self, activation=None):
        super(LinearModelForPrunning, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=3, out_features=10)
        self.fc2 = torch.nn.Linear(in_features=10, out_features=6)
        self.fc3 = torch.nn.Linear(in_features=6, out_features=6)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        if self.activation:
            x = self.activation(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class LinearPruningTest(PruningPytorchFeatureTest):
    """
    Test a network with two adjacent conv2d and check it's pruned a single group of channels.
    """

    def __init__(self,
                 unit_test,
                 activation_layer=None,
                 simd=1,
                 use_constant_importance_metric=True):

        super().__init__(unit_test,
                         input_shape=(64, 3))
        self.activation_layer = activation_layer
        self.simd = simd
        self.use_constant_importance_metric = use_constant_importance_metric


    def get_pruning_config(self):
        if self.use_constant_importance_metric:
            add_const_importance_metric(first_num_oc=10, second_num_oc=6, simd=self.simd)
            return mct.pruning.PruningConfig(importance_metric=ConstImportanceMetric.CONST)
        return super().get_pruning_config()

    def create_networks(self):
        return LinearModelForPrunning(activation=self.activation_layer)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        dense_layers = get_layers_from_model_by_type(float_model, torch.nn.Linear)
        prunable_layers = get_layers_from_model_by_type(quantized_model, torch.nn.Linear)

        is_first_layer_pruned = prunable_layers[0].out_features == dense_layers[0].out_features - self.simd
        is_second_layer_pruned = prunable_layers[1].out_features == dense_layers[1].out_features - self.simd

        # Make sure only one of layers has been pruned
        self.unit_test.assertTrue(is_first_layer_pruned != is_second_layer_pruned)

        # In constant case, the last SIMD channels of the first layer should be pruned:
        if self.use_constant_importance_metric:
            self.unit_test.assertTrue(is_first_layer_pruned)
            self.unit_test.assertTrue(np.all(torch_tensor_to_numpy(prunable_layers[0].weight)==torch_tensor_to_numpy(dense_layers[0].weight)[:-self.simd, :]))
            self.unit_test.assertTrue(np.all(torch_tensor_to_numpy(prunable_layers[0].bias)==torch_tensor_to_numpy(dense_layers[0].bias)[:-self.simd]))
            self.unit_test.assertTrue(np.all(torch_tensor_to_numpy(prunable_layers[1].weight) == torch_tensor_to_numpy(dense_layers[1].weight)[:, :-self.simd]))
            self.unit_test.assertTrue(np.all(torch_tensor_to_numpy(prunable_layers[1].bias)==torch_tensor_to_numpy(dense_layers[1].bias)))

        if is_first_layer_pruned:
            self.unit_test.assertTrue(np.all(torch_tensor_to_numpy(prunable_layers[2].weight) == torch_tensor_to_numpy(dense_layers[2].weight)))
            self.unit_test.assertTrue(np.all(torch_tensor_to_numpy(prunable_layers[2].bias) == torch_tensor_to_numpy(dense_layers[2].bias)))


