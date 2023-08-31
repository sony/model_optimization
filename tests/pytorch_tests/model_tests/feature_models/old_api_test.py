# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import numpy as np

import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tp_model import generate_tp_model
from tests.common_tests.helpers.generate_test_tp_model import generate_mixed_precision_test_tp_model
from tests.pytorch_tests.model_tests.base_pytorch_test import BasePytorchTest
from tests.pytorch_tests.tpc_pytorch import get_pytorch_test_tpc_dict
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import get_tp_model, generate_pytorch_tpc, get_op_quantization_configs


class OneLayerConv2dNet(torch.nn.Module):
    def __init__(self):
        super(OneLayerConv2dNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class OldApiTest(BasePytorchTest):
    def __init__(self, unit_test, mp_enable=False, gptq_enable=False):
        super().__init__(unit_test)
        self.num_calibration_iter = 100
        self.mp_enable = mp_enable
        self.gptq_enable = gptq_enable
        self.input_shape = (1, 3, 8, 8)

    def get_mp_tpc(self):
        base_config, _ = get_op_quantization_configs()
        base_config = base_config.clone_and_edit(weights_n_bits=16,
                                                 activation_n_bits=16)
        mp_bitwidth_candidates_list = [(8, 16), (2, 16), (4, 16), (16, 16)]
        mp_op_cfg_list = []
        for weights_n_bits, activation_n_bits in mp_bitwidth_candidates_list:
            candidate_cfg = base_config.clone_and_edit(weights_n_bits=weights_n_bits,
                                                       activation_n_bits=activation_n_bits)
            mp_op_cfg_list.append(candidate_cfg)

        tp_model = generate_tp_model(default_config=base_config,
                                     base_config=base_config,
                                     mixed_precision_cfg_list=mp_op_cfg_list,
                                     name='default_tp_model')
        return get_pytorch_test_tpc_dict(tp_model=tp_model,
                                         test_name='mixed_precision_model',
                                         ftp_name='mixed_precision_pytorch_test')

    def get_mp_quant_config(self):
        qc = mct.core.QuantizationConfig(mct.core.QuantizationErrorMethod.MSE,
                                    mct.core.QuantizationErrorMethod.MSE,
                                    weights_bias_correction=True,
                                    weights_per_channel_threshold=True,
                                    activation_channel_equalization=False,
                                    relu_bound_to_power_of_2=False,
                                    input_scaling=False)
        return mct.core.MixedPrecisionQuantizationConfig(qc, num_of_images=1)

    def get_kpi(self):
        return mct.core.KPI()

    def create_networks(self):
        return OneLayerConv2dNet()

    def get_gptq_config(self):
        return mct.gptq.GradientPTQConfig(5, optimizer=torch.optim.Adam([torch.Tensor([])], lr=1e-14))

    def generate_inputs(self):
        return to_torch_tensor([torch.randn(*self.input_shape) for in_shape in self.input_shape])

    def run_test(self, experimental_exporter=False):
        model_float = self.create_networks()
        core_config = self.get_core_config()
        quant_config = core_config.quantization_config
        gptq_config = self.get_gptq_config() if self.gptq_enable else None
        if self.mp_enable:
            quant_config = self.get_mp_quant_config()
            facade_fn = mct.pytorch_post_training_quantization_mixed_precision
            ptq_model, quantization_info = facade_fn(model_float,
                                                     self.representative_data_gen,
                                                     self.get_kpi(),
                                                     n_iter=self.num_calibration_iter,
                                                     quant_config=quant_config,
                                                     gptq_config=gptq_config,
                                                     target_platform_capabilities=self.get_mp_tpc()['mixed_precision_model'],
                                                     )
        else:
            facade_fn = mct.pytorch_post_training_quantization
            ptq_model, quantization_info = facade_fn(model_float,
                                                     self.representative_data_gen,
                                                     n_iter=self.num_calibration_iter,
                                                     quant_config=quant_config,
                                                     gptq_config=gptq_config,
                                                     target_platform_capabilities=self.get_tpc()['all_32bit'],
                                                     )

        self.compare(ptq_model, model_float, input_x=self.representative_data_gen(),
                     quantization_info=quantization_info)

    def compare(self, quant_model, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        out_float = torch_tensor_to_numpy(float_model(input_x[0]))
        out_quant = torch_tensor_to_numpy(quant_model(input_x[0]))
        self.unit_test.assertTrue(np.isclose(np.linalg.norm(np.abs(out_float-out_quant)), 0, atol=0.01))
