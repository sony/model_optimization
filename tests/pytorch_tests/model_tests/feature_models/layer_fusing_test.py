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
import torch.nn as nn
from torch.nn import Conv2d, ReLU, SiLU, Sigmoid, Linear, Hardtanh
from torch.nn.functional import relu, relu6
import model_compression_toolkit as mct
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
from model_compression_toolkit.core.common.target_platform.targetplatform2framework.layer_filter_params import LayerFilterParams

tp = mct.target_platform


class BaseLayerFusingTest(BasePytorchFeatureNetworkTest):

    def __init__(self, unit_test):
        super().__init__(unit_test=unit_test, input_shape=(3, 16, 16))
        self.expected_fusions = []

    def get_type(self, fusion):
        fusion_types = [x.type for x in fusion]
        return fusion_types

    def get_tpc(self):
        default_config, mixed_precision_cfg_list = get_op_quantization_configs()
        default_configuration_options = tp.QuantizationConfigOptions([default_config])
        generated_tp = tp.TargetPlatformModel(default_configuration_options, name='layer_fusing_test')
        mixed_precision_configuration_options = tp.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                             base_config=default_config)
        return generated_tp, mixed_precision_configuration_options

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantization_info.fusions) == len(self.expected_fusions), msg=f'Number of fusions is not as expected!')
        for i,fusion in enumerate(quantization_info.fusions):
            self.unit_test.assertTrue(self.get_type(fusion) == self.expected_fusions[i], msg=f'Miss-match fusion compared to expected!')

class LayerFusingTest1(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d,ReLU]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            any_relu = tp.OperatorsSet("AnyReLU")
            # Define fusions
            tp.Fusing([conv, any_relu])

        pytorch_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with pytorch_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2d])
            tp.OperationsSetToLayers("AnyReLU", [torch.relu,
                                                 ReLU])
        return pytorch_tpc

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=(3,3))
            self.conv2 = nn.Conv2d(16, 32, kernel_size=(1,1))
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            y = self.relu(x)
            return y

    def create_networks(self):
        return self.LayerFusingNetTest()


class LayerFusingTest2(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d, Hardtanh], [Conv2d, ReLU], [Conv2d, Sigmoid], [Conv2d, SiLU]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            any_act = tp.OperatorsSet("AnyAct")
            # Define fusions
            tp.Fusing([conv, any_act])

        pytorch_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with pytorch_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2d])
            tp.OperationsSetToLayers("AnyAct", [ReLU,relu6,relu,SiLU,Sigmoid,LayerFilterParams(Hardtanh, min_val=0)])
        return pytorch_tpc

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,1))
            self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3))
            self.conv4 = nn.Conv2d(32, 64, kernel_size=(1,1))
            self.conv5 = nn.Conv2d(64, 64, kernel_size=(2,2))
            self.relu = nn.ReLU()
            self.tanh = Hardtanh(min_val=0)
            self.swish = nn.SiLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.tanh(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.sigmoid(x)
            x = self.conv5(x)
            y = self.swish(x)
            return y

    def create_networks(self):
        return self.LayerFusingNetTest()


class LayerFusingTest3(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d, ReLU]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            any_act = tp.OperatorsSet("AnyAct")
            # Define fusions
            tp.Fusing([conv, any_act])

        pytorch_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with pytorch_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2d])
            tp.OperationsSetToLayers("AnyAct", [ReLU,relu6,relu])
        return pytorch_tpc

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3))
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,1))
            self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3))
            self.conv4 = nn.Conv2d(32, 64, kernel_size=(1,1))
            self.conv5 = nn.Conv2d(64, 64, kernel_size=(2,2))
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.swish = nn.SiLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.tanh(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.sigmoid(x)
            x = self.conv5(x)
            y = self.swish(x)
            return y

    def create_networks(self):
        return self.LayerFusingNetTest()


class LayerFusingTest4(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2d, SiLU, torch.add], [Conv2d, SiLU, torch.add], [Conv2d, ReLU], [Conv2d, ReLU, torch.add], [Linear, SiLU], [Linear, SiLU]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            fc = tp.OperatorsSet("FullyConnected", mixed_precision_configuration_options)
            any_relu = tp.OperatorsSet("AnyReLU")
            add = tp.OperatorsSet("Add")
            swish = tp.OperatorsSet("Swish")
            activations_to_fuse = tp.OperatorSetConcat(any_relu, swish)
            # Define fusions
            tp.Fusing([conv, activations_to_fuse])
            tp.Fusing([conv, add, activations_to_fuse])
            tp.Fusing([conv, activations_to_fuse, add])
            tp.Fusing([fc, activations_to_fuse])

        pytorch_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with pytorch_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2d])
            tp.OperationsSetToLayers("FullyConnected", [Linear])
            tp.OperationsSetToLayers("AnyReLU", [ReLU])
            tp.OperationsSetToLayers("Add", [torch.add])
            tp.OperationsSetToLayers("Swish", [SiLU])
        return pytorch_tpc

    class LayerFusingNetTest(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 3, kernel_size=(3,3), padding='same')
            self.conv2 = nn.Conv2d(3, 3, kernel_size=(1,1), padding='same')
            self.conv3 = nn.Conv2d(3, 3, kernel_size=(3,3), padding='same')
            self.conv4 = nn.Conv2d(3, 3, kernel_size=(1,1), padding='same')
            self.conv5 = nn.Conv2d(3, 3, kernel_size=(3,3), padding='same')
            self.conv6 = nn.Conv2d(3, 3, kernel_size=(1,1), padding='same')
            self.relu = nn.ReLU()
            self.swish = nn.SiLU()
            self.flatten = nn.Flatten()
            self.dense1 = nn.Linear(768, out_features=16)
            self.dense2 = nn.Linear(16, out_features=16)

        def forward(self, inputs):
            x = self.conv1(inputs)
            x = self.swish(x)
            x1 = torch.add(inputs, x)
            x2 = self.conv2(x1)
            x2 = self.swish(x2)
            x2 = torch.add(x1, x2)
            x2 = self.conv3(x2)
            x2 = self.relu(x2)
            x3 = self.conv4(x2)
            x3 = self.relu(x3)
            x3 = torch.add(x3, x2)
            x3 = self.flatten(x3)
            x3 = self.dense1(x3)
            x3 = self.swish(x3)
            x3 = self.dense2(x3)
            y = self.swish(x3)
            return y

    def create_networks(self):
        return self.LayerFusingNetTest()

