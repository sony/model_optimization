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

from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, GradientPTQConfigV2, RoundingType
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy
from model_compression_toolkit.gptq.pytorch.gptq_loss import multiple_tensors_mse_loss


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
    def __init__(self, unit_test, experimental_exporter=False, rounding_type=RoundingType.STE, per_channel=True):
        super().__init__(unit_test, input_shape=(3, 16, 16))
        self.seed = 0
        self.experimental = True
        self.experimental_exporter = experimental_exporter
        self.rounding_type = rounding_type
        self.per_channel = per_channel

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                      mct.QuantizationErrorMethod.NOCLIPPING,
                                      weights_per_channel_threshold=self.per_channel)

    def create_networks(self):
        return TestModel()

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        pass

    def run_test(self):
        # Create model
        self.float_model = self.create_networks()

        # Create quantization config
        qConfig = self.get_quantization_config()

        # Run MCT with PTQ
        np.random.seed(self.seed)
        ptq_model, _ = mct.pytorch_post_training_quantization_experimental(self.float_model,
                                                                           self.representative_data_gen_experimental,
                                                                           core_config=self.get_core_config(),
                                                                           target_platform_capabilities=self.get_tpc()) if self.experimental \
            else mct.pytorch_post_training_quantization(self.float_model,
                                                        self.representative_data_gen,
                                                        n_iter=self.num_calibration_iter,
                                                        quant_config=qConfig,
                                                        fw_info=DEFAULT_PYTORCH_INFO,
                                                        network_editor=self.get_network_editor())

        # Run MCT with GPTQ
        np.random.seed(self.seed)
        gptq_model, quantization_info = mct.pytorch_gradient_post_training_quantization_experimental(self.float_model,
                                                                                                     self.representative_data_gen_experimental,
                                                                                                     core_config=self.get_core_config(),
                                                                                                     target_platform_capabilities=self.get_tpc(),
                                                                                                     gptq_config=self.get_gptq_configv2(),
                                                                                                     new_experimental_exporter=self.experimental_exporter) if self.experimental \
            else mct.pytorch_post_training_quantization(self.float_model,
                                                        self.representative_data_gen,
                                                        n_iter=self.num_calibration_iter,
                                                        quant_config=qConfig,
                                                        fw_info=DEFAULT_PYTORCH_INFO,
                                                        network_editor=self.get_network_editor(),
                                                        gptq_config=self.get_gptq_config())

        # Generate inputs
        x = to_torch_tensor(self.representative_data_gen())

        # Compare
        self.gptq_compare(ptq_model, gptq_model, input_x=x)


class GPTQAccuracyTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(5,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=False,
                                 use_jac_based_weights=True,
                                 optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                 rounding_type=self.rounding_type)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=False,
                                   use_jac_based_weights=True,
                                   optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                   rounding_type=self.rounding_type)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')


class GPTQWeightsUpdateTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(50,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=True,
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 rounding_type=self.rounding_type)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(50,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=True,
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   rounding_type=self.rounding_type)

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
        return GradientPTQConfig(5,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=False,
                                 rounding_type=self.rounding_type)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=False,
                                   rounding_type=self.rounding_type)

    def compare(self, ptq_model, gptq_model, input_x=None, quantization_info=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were not updated in gptq model compared to ptq model
        w_diffs = [np.isclose(np.max(np.abs(w_ptq - w_gptq)), 0) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        for w_diff in w_diffs:
            self.unit_test.assertTrue(np.all(w_diff), msg="GPTQ: some weights were updated")

