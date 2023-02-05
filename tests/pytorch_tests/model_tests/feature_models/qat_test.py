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
from torch import Tensor

from model_compression_toolkit.core.pytorch.utils import get_working_device
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc
from model_compression_toolkit import quantizers_infrastructure as qi


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1))
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
    def __init__(self, unit_test, weight_bits=2, activation_bits=4,
                 weights_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 activation_quantization_method=mct.target_platform.QuantizationMethod.POWER_OF_TWO,
                 finalize=False, test_loading=False):

        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.finalize = finalize
        self.weights_quantization_method = weights_quantization_method
        self.activation_quantization_method = activation_quantization_method
        self.test_loading = test_loading
        super().__init__(unit_test, input_shape=(3, 4, 4))

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="qat_test",
            tp_model=generate_test_tp_model({'weights_n_bits': self.weight_bits,
                                             'activation_n_bits': self.activation_bits,
                                             'weights_quantization_method': self.weights_quantization_method,
                                             'activation_quantization_method': self.activation_quantization_method}))

    def create_networks(self):
        return TestModel()

    def _gen_fixed_input(self):
        self.fixed_inputs = self.generate_inputs()

    def representative_data_gen_experimental(self):
        for _ in range(self.num_calibration_iter):
            yield self.fixed_inputs

    def run_test(self, experimental_facade=False):
        self._gen_fixed_input()
        model_float = self.create_networks()
        _tpc = self.get_tpc()
        ptq_model, quantization_info = mct.pytorch_post_training_quantization_experimental(model_float,
                                                                                           self.representative_data_gen_experimental,
                                                                                           target_platform_capabilities=_tpc)

        qat_ready_model, quantization_info = mct.pytorch_quantization_aware_training_init(model_float,
                                                                                          self.representative_data_gen_experimental,
                                                                                          target_platform_capabilities=_tpc)

        if self.test_loading:
            pass # TODO: need to save and load pytorch model

        if self.finalize:
            qat_finalized_model = mct.pytorch_quantization_aware_training_finalize(qat_ready_model)
        else:
            qat_finalized_model = None

        self.compare(ptq_model,
                     qat_ready_model,
                     qat_finalized_model,
                     input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, ptq_model, qat_ready_model, qat_finalized_model, input_x=None, quantization_info=None):
        # check relevant layers are wrapped and correct quantizers were chosen
        for _, layer in qat_ready_model.named_children():
            self.unit_test.assertTrue(isinstance(layer, qi.PytorchQuantizationWrapper))
            # if isinstance(layer.layer, nn.SiLU):
                # q = METHOD2ACTQUANTIZER[mct.TrainingMethod.STE][self.activation_quantization_method]
                # self.unit_test.assertTrue(isinstance(layer.activation_quantizers[0], q))
            # if isinstance(layer.layer, nn.Conv2d):
                # q = METHOD2WEIGHTQUANTIZER[mct.TrainingMethod.STE][self.activation_quantization_method]
                # self.unit_test.assertTrue(isinstance(layer.weights_quantizers['weight'], q))

        # check quantization didn't change when switching between PTQ model and QAT ready model
        _in = Tensor(input_x[0]).to(get_working_device())
        ptq_output = ptq_model(_in).cpu().detach().numpy()
        qat_ready_output = qat_ready_model(_in).cpu().detach().numpy()
        self.unit_test.assertTrue(np.isclose(np.linalg.norm(ptq_output - qat_ready_output) / np.linalg.norm(ptq_output), 0, atol=1e-6))
        if self.finalize:
            for _, layer in qat_finalized_model.named_children():
                self.unit_test.assertTrue(isinstance(layer, qi.PytorchQuantizationWrapper))
                # if isinstance(layer.layer, nn.SiLU):
                #     q = QUANTIZATION_METHOD_2_ACTIVATION_QUANTIZER[self.activation_quantization_method]
                #     self.unit_test.assertTrue(isinstance(layer.activation_quantizers[0], q))
                # if isinstance(layer.layer, nn.Conv2d):
                #     q = QUANTIZATION_METHOD_2_WEIGHTS_QUANTIZER[self.activation_quantization_method]
                #     self.unit_test.assertTrue(isinstance(layer.weights_quantizers['weight'], q))
            # check quantization didn't change when switching between PTQ model and QAT ready model
            qat_finalized_output = qat_finalized_model(_in).cpu().detach().numpy()
            self.unit_test.assertTrue(np.isclose(np.linalg.norm(qat_finalized_output - qat_ready_output) / np.linalg.norm(qat_ready_output), 0, atol=1e-6))
