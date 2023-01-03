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


import numpy as np
import torch.nn as nn

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit import qunatizers_infrastructure as qi
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
        self.activation = nn.SiLU()

    def forward(self, inp):
        x0 = self.conv1(inp)
        x1 = self.activation(x0)
        x2 = self.conv2(x1)
        y = self.activation(x2)
        return y


def repr_datagen():
    for _ in range(10):
        yield [np.random.random((4, 3, 224, 224))]


class QuantizationAwareTrainingTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, weight_bits=2, activation_bits=4, finalize=False,
                 weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO, test_loading=False):

        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.weights_quantization_method = weights_quantization_method
        self.test_loading = test_loading
        super().__init__(unit_test)

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="qat_test",
            tp_model=generate_test_tp_model({"weights_quantization_method": self.weights_quantization_method}))

    def create_networks(self):
        return TestModel()

    def run_test(self, experimental_facade=False):
        model_float = self.create_networks()
        ptq_model, quantization_info = mct.pytorch_quantization_aware_training_init(model_float,
                                                                                    self.representative_data_gen_experimental,
                                                                                    fw_info=self.get_fw_info(),
                                                                                    target_platform_capabilities=self.get_tpc())

        ptq_model2 = ptq_model
        if self.test_loading:
            pass # TODO: need to save and load pytorch model

        if self.finalize:
            ptq_model = mct.pytorch_quantization_aware_training_finalize(ptq_model)

        self.compare(ptq_model,
                     model_float,
                     ptq_model2,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quantized_model, float_model, loaded_model, input_x=None, quantization_info=None):
        for layer in loaded_model.named_children():
            if not layer[0] == 'inp' and not 'activation' in layer[0]:
                self.unit_test.assertTrue(isinstance(layer[1], qi.PytorchQuantizationWrapper))


class QATSymmetricActivationTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, weight_bits=8, activation_bits=8, finalize=False,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.SYMMETRIC):

        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.activation_quantization_method = activation_quantization_method
        super().__init__(unit_test)

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="qat_activation_test",
            tp_model=generate_test_tp_model({"activation_quantization_method": self.activation_quantization_method}))

    def create_networks(self):
        return TestModel()

    def run_test(self, experimental_facade=False):
        float_model = self.create_networks()
        qat_model, quantization_info = mct.pytorch_quantization_aware_training_init(float_model,
                                                                                    self.representative_data_gen_experimental,
                                                                                    fw_info=self.get_fw_info(),
                                                                                    target_platform_capabilities=self.get_tpc())

        # ----------- #
        #  QAT stage  #
        # ----------- #
        x = to_torch_tensor(np.random.random((4, 3, 224, 224)))
        qat_model(x)
        # -----------#

        exported_model = qat_model
        if self.finalize:
            exported_model = mct.pytorch_quantization_aware_training_finalize(qat_model)

        self.compare(qat_model,
                     float_model,
                     exported_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quantized_model, float_model, exported_model, input_x=None, quantization_info=None):
        # Check that all layers are wrapped with PytorchQuantizationWrapper
        for name, layer in exported_model.named_children():
            self.unit_test.assertTrue(isinstance(layer, qi.PytorchQuantizationWrapper))


class QATUniformActivationTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, weight_bits=8, activation_bits=8, finalize=False,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.UNIFORM):

        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.activation_quantization_method = activation_quantization_method
        super().__init__(unit_test)

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="qat_activation_test",
            tp_model=generate_test_tp_model({"activation_quantization_method": self.activation_quantization_method}))

    def create_networks(self):
        return TestModel()

    def run_test(self, experimental_facade=False):
        float_model = self.create_networks()
        qat_model, quantization_info = mct.pytorch_quantization_aware_training_init(float_model,
                                                                                    self.representative_data_gen_experimental,
                                                                                    fw_info=self.get_fw_info(),
                                                                                    target_platform_capabilities=self.get_tpc())

        # ----------- #
        #  QAT stage  #
        # ----------- #
        x = to_torch_tensor(np.random.random((4, 3, 224, 224)))
        qat_model(x)
        # -----------#

        exported_model = qat_model
        if self.finalize:
            exported_model = mct.pytorch_quantization_aware_training_finalize(qat_model)

        self.compare(qat_model,
                     float_model,
                     exported_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quantized_model, float_model, exported_model, input_x=None, quantization_info=None):
        # Check that all layers are wrapped with PytorchQuantizationWrapper
        for name, layer in exported_model.named_children():
            self.unit_test.assertTrue(isinstance(layer, qi.PytorchQuantizationWrapper))
