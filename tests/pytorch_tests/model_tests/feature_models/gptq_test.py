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
import torch
import torch.nn as nn

from model_compression_toolkit.core import DefaultDict
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod
from model_compression_toolkit.gptq.common.gptq_constants import QUANT_PARAM_LEARNING_STR, MAX_LSB_STR
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, GradientPTQConfigV2, RoundingType, \
    GPTQHessianWeightsConfig
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from model_compression_toolkit.gptq.pytorch.gptq_loss import multiple_tensors_mse_loss
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model

tp = mct.target_platform


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
        self.activation = nn.SiLU()
        self.linear = nn.Linear(14, 3)

    def forward(self, inp):
        x0 = self.conv1(inp)
        x1 = self.activation(x0)
        x2 = self.conv2(x1)
        y = self.activation(x2)
        y = self.linear(y)
        return y


class GPTQBaseTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, experimental_exporter=True, weights_bits=8, weights_quant_method=QuantizationMethod.SYMMETRIC,
                 rounding_type=RoundingType.STE, per_channel=True,
                 hessian_weights=True, log_norm_weights=True, scaled_log_norm=False, params_learning=True):
        super().__init__(unit_test, input_shape=(3, 16, 16))
        self.seed = 0
        self.experimental_exporter = experimental_exporter
        self.rounding_type = rounding_type
        self.weights_bits = weights_bits
        self.weights_quant_method = weights_quant_method
        self.per_channel = per_channel
        self.hessian_weights = hessian_weights
        self.log_norm_weights = log_norm_weights
        self.scaled_log_norm = scaled_log_norm
        self.override_params = {QUANT_PARAM_LEARNING_STR: params_learning} if \
            rounding_type == RoundingType.SoftQuantizer else {MAX_LSB_STR: DefaultDict({}, lambda: 1)} \
            if rounding_type == RoundingType.STE else None

    def get_quantization_config(self):
        return mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.NOCLIPPING,
                                      mct.core.QuantizationErrorMethod.NOCLIPPING,
                                      weights_per_channel_threshold=self.per_channel)

    def create_networks(self):
        return TestModel()

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="gptq_test",
            tp_model=generate_test_tp_model({'weights_n_bits': self.weights_bits,
                                             'weights_quantization_method': self.weights_quant_method}))

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        pass

    def run_test(self):
        # Create model
        self.float_model = self.create_networks()
        set_model(self.float_model)

        # Run MCT with PTQ
        np.random.seed(self.seed)
        ptq_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(self.float_model,
                                                                           self.representative_data_gen_experimental,
                                                                           core_config=self.get_core_config(),
                                                                           target_platform_capabilities=self.get_tpc(),
                                                                            new_experimental_exporter=False)

        # Run MCT with GPTQ
        np.random.seed(self.seed)
        gptq_model, quantization_info = mct.gptq.pytorch_gradient_post_training_quantization_experimental(
            self.float_model,
            self.representative_data_gen_experimental,
            core_config=self.get_core_config(),
            target_platform_capabilities=self.get_tpc(),
            gptq_config=self.get_gptq_configv2(),
            new_experimental_exporter=self.experimental_exporter)

        # Generate inputs
        x = to_torch_tensor(self.representative_data_gen())

        # Compare
        self.gptq_compare(ptq_model, gptq_model, input_x=x)


class GPTQAccuracyTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 loss=multiple_tensors_mse_loss, train_bias=True, rounding_type=self.rounding_type,
                                 use_hessian_based_weights=self.hessian_weights,
                                 optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                 hessian_weights_config=GPTQHessianWeightsConfig(log_norm=self.log_norm_weights,
                                                                                 scale_log_norm=self.scaled_log_norm),
                                 gptq_quantizer_params_override=self.override_params)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   loss=multiple_tensors_mse_loss, train_bias=True, rounding_type=self.rounding_type,
                                   use_hessian_based_weights=self.hessian_weights,
                                   optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                   hessian_weights_config=GPTQHessianWeightsConfig(log_norm=self.log_norm_weights,
                                                                                   scale_log_norm=self.scaled_log_norm),
                                   gptq_quantizer_params_override=self.override_params)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')


class GPTQWeightsUpdateTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(50, optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 loss=multiple_tensors_mse_loss, train_bias=True, rounding_type=self.rounding_type,
                                 gptq_quantizer_params_override=self.override_params)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(50, optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   loss=multiple_tensors_mse_loss, train_bias=True, rounding_type=self.rounding_type,
                                   gptq_quantizer_params_override=self.override_params)

    def compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were updated in gptq model compared to ptq model
        w_diff = [np.any(w_ptq != w_gptq) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(all(w_diff), msg="GPTQ: some weights weren't updated")


class GPTQLearnRateZeroTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0),
                                 loss=multiple_tensors_mse_loss, train_bias=False, rounding_type=self.rounding_type,
                                 gptq_quantizer_params_override=self.override_params)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0),
                                   loss=multiple_tensors_mse_loss, train_bias=False, rounding_type=self.rounding_type,
                                   gptq_quantizer_params_override=self.override_params)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        ptq_out = torch_tensor_to_numpy(ptq_model(input_x))
        gptq_out = torch_tensor_to_numpy(gptq_model(input_x))
        float_output = torch_tensor_to_numpy(self.float_model(torch.Tensor(input_x[0])))
        self.unit_test.assertTrue(np.isclose(np.linalg.norm(ptq_out - float_output),
                                             np.linalg.norm(gptq_out - float_output), atol=1e-3))

        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were not updated in gptq model compared to ptq model
        w_diffs = [np.isclose(np.max(np.abs(w_ptq - w_gptq)), 0) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(np.all(w_diffs), msg="GPTQ: some weights were updated in zero learning rate test")
