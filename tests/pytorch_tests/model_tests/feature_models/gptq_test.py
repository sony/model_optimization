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
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.core.tpc_models.default_tpc.latest import generate_pytorch_tpc

tp = mct.target_platform


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


class GPTQBaseTest(BasePytorchFeatureNetworkTest):
    def __init__(self, unit_test, experimental_exporter=False):
        super().__init__(unit_test, input_shape=(3, 16, 16))
        self.seed = 0
        self.experimental = True
        self.experimental_exporter = experimental_exporter

    def get_quantization_config(self):
        return mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING, mct.QuantizationErrorMethod.NOCLIPPING)

    def create_networks(self):
        return TestModel()

    def gptq_compare(self, ptq_model, gptq_model, input_x=None, max_change=0):
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
        bits = self.get_tpc().tp_model.default_qco.quantization_config_list[0].weights_n_bits
        max_change = self.get_gptq_config().lsb_change_per_bit_width.get(bits)/(2**(bits-1))
        self.gptq_compare(ptq_model, gptq_model, input_x=x, max_change=max_change)


class STEAccuracyTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(5,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=False,
                                 use_jac_based_weights=True,
                                 optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                 rounding_type=RoundingType.STE)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=False,
                                   use_jac_based_weights=True,
                                   optimizer_bias=torch.optim.Adam([torch.Tensor([])], lr=0.4),
                                   rounding_type=RoundingType.STE)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        for ptq_param, gptq_param in zip(ptq_model.parameters(), gptq_model.parameters()):
            delta_w = torch.abs(gptq_param - ptq_param)
            if len(delta_w.shape) > 1: # Skip bias
                self.unit_test.assertTrue(torch.max(delta_w) <= max_change, msg='GPTQ weights changes are bigger than expected!')


class STEWeightsUpdateTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(50,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=True,
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 rounding_type=RoundingType.STE)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(50,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=True,
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   rounding_type=RoundingType.STE)

    def compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were updated in gptq model compared to ptq model
        w_diff = [np.any(w_ptq != w_gptq) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(all(w_diff), msg="GPTQ: some weights weren't updated")


class STELearnRateZeroTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(5,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=False,
                                 rounding_type=RoundingType.STE)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=0),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=False,
                                   rounding_type=RoundingType.STE)

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


class SymGumbelAccuracyTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(5,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=False,
                                 use_jac_based_weights=True,
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 quantization_parameters_learning=True,
                                 rounding_type=RoundingType.GumbelRounding)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=False,
                                   use_jac_based_weights=True,
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   quantization_parameters_learning=True,
                                   rounding_type=RoundingType.GumbelRounding)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        for ptq_param, gptq_param in zip(ptq_model.parameters(), gptq_model.parameters()):
            delta_w = torch.abs(gptq_param - ptq_param)
            if len(delta_w.shape) > 1: # Skip bias
                self.unit_test.assertTrue(torch.max(delta_w) <= max_change, msg='GPTQ weights changes are bigger than expected!')


class SymGumbelWeightsUpdateTest(GPTQBaseTest):

    def get_gptq_config(self):
        return GradientPTQConfig(50,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=True,
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 rounding_type=RoundingType.GumbelRounding)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(50,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=True,
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   rounding_type=RoundingType.GumbelRounding)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were updated in gptq model compared to ptq model
        w_diff = [np.any(w_ptq != w_gptq) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(all(w_diff), msg="GPTQ: some weights weren't updated")


class UniformGumbelAccuracyTest(GPTQBaseTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="gptq_uniform_gumbel_test",
            tp_model=generate_test_tp_model({"weights_quantization_method": tp.QuantizationMethod.UNIFORM}))

    def get_gptq_config(self):
        return GradientPTQConfig(5,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=False,
                                 use_jac_based_weights=True,
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                 quantization_parameters_learning=True,
                                 rounding_type=RoundingType.GumbelRounding)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(5,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=False,
                                   use_jac_based_weights=True,
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=1e-4),
                                   quantization_parameters_learning=True,
                                   rounding_type=RoundingType.GumbelRounding)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        for ptq_param, gptq_param in zip(ptq_model.parameters(), gptq_model.parameters()):
            delta_w = torch.abs(gptq_param - ptq_param)
            if len(delta_w.shape) > 1: # Skip bias
                self.unit_test.assertTrue(torch.max(delta_w) <= max_change, msg='GPTQ weights changes are bigger than expected!')


class UniformGumbelWeightsUpdateTest(GPTQBaseTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_tpc(self):
        return generate_pytorch_tpc(
            name="gptq_uniform_gumbel_test",
            tp_model=generate_test_tp_model({"weights_quantization_method": tp.QuantizationMethod.UNIFORM}))

    def get_gptq_config(self):
        return GradientPTQConfig(50,
                                 optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 loss=multiple_tensors_mse_loss,
                                 train_bias=True,
                                 optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                 rounding_type=RoundingType.GumbelRounding)

    def get_gptq_configv2(self):
        return GradientPTQConfigV2(50,
                                   optimizer=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   loss=multiple_tensors_mse_loss,
                                   train_bias=True,
                                   optimizer_rest=torch.optim.Adam([torch.Tensor([])], lr=0.5),
                                   rounding_type=RoundingType.GumbelRounding)

    def gptq_compare(self, ptq_model, gptq_model, input_x=None, max_change=None):
        ptq_weights = torch_tensor_to_numpy(list(ptq_model.parameters()))
        gptq_weights = torch_tensor_to_numpy(list(gptq_model.parameters()))

        # check number of weights are equal
        self.unit_test.assertTrue(len(ptq_weights) == len(gptq_weights),
                                  msg='PTQ model number of weights different from GPTQ model!')

        # check all weights were updated in gptq model compared to ptq model
        w_diff = [np.any(w_ptq != w_gptq) for w_ptq, w_gptq in zip(ptq_weights, gptq_weights)]
        self.unit_test.assertTrue(all(w_diff), msg="GPTQ: some weights weren't updated")
