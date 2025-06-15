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
import random
from torch.fx import symbolic_trace

from model_compression_toolkit.core.pytorch.utils import set_model, to_torch_tensor, \
    torch_tensor_to_numpy
import model_compression_toolkit as mct
import torch
import numpy as np

from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.pytorch.default_framework_info import PyTorchInfo
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest
from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc

"""
The base test class for the feature networks
"""


class BasePytorchTest(BaseFeatureNetworkTest):
    def __init__(self,
                 unit_test,
                 float_reconstruction_error=1e-6,
                 convert_to_fx=True,
                 val_batch_size=1,
                 num_calibration_iter=1):

        super().__init__(unit_test, val_batch_size=val_batch_size, num_calibration_iter=num_calibration_iter)
        self.float_reconstruction_error = float_reconstruction_error
        self.convert_to_fx = convert_to_fx
        set_fw_info(PyTorchInfo)

    def get_tpc(self):
        return {
            'no_quantization': generate_test_tpc({'weights_n_bits': 32,
                                                       'activation_n_bits': 32,
                                                       'enable_weights_quantization': False,
                                                       'enable_activation_quantization': False
                                                  }),
            'all_32bit': generate_test_tpc({'weights_n_bits': 32,
                                                 'activation_n_bits': 32,
                                                 'enable_weights_quantization': True,
                                                 'enable_activation_quantization': True
                                            }),
            'all_4bit': generate_test_tpc({'weights_n_bits': 4,
                                                'activation_n_bits': 4,
                                                'enable_weights_quantization': True,
                                                'enable_activation_quantization': True
                                           }),
        }

    def get_core_configs(self):
        base_core_config = mct.core.CoreConfig(quantization_config=self.get_quantization_config(),
                                               mixed_precision_config=self.get_mixed_precision_config(),
                                               debug_config=self.get_debug_config())
        return {
            'no_quantization': base_core_config,
            'all_32bit': base_core_config,
            'all_4bit': base_core_config,
        }

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def create_feature_network(self, input_shape):
        pass

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None):
        set_model(float_model)
        float_result = float_model(*input_x)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quant_result = quantized_model(*input_x)
            for i, (f, q) in enumerate(zip(float_result, quant_result)):
                if model_name == 'no_quantization':
                    # Check if we have a BatchNorm or MultiheadAttention layer in the model.
                    # If so, the outputs will not be the same, since the sqrt function in the
                    # Decomposition is not exactly like the sqrt in the C implementation of PyTorch.
                    float_model_operators = [type(module) for name, module in float_model.named_modules()]
                    if (torch.nn.BatchNorm2d in float_model_operators or
                        torch.nn.MultiheadAttention in float_model_operators or self.use_is_close_validation):
                         self.unit_test.assertTrue(np.all(np.isclose(torch_tensor_to_numpy(f), torch_tensor_to_numpy(q),
                                                                    atol=self.float_reconstruction_error)))
                    else:
                        self.unit_test.assertTrue(torch_tensor_to_numpy(torch.sum(torch.abs(f - q))) == 0,
                                                  msg=f'observed distance: {torch_tensor_to_numpy(torch.sum(torch.abs(f - q)))} should be zero')
                elif model_name == 'all_4bit':
                    self.unit_test.assertFalse(torch_tensor_to_numpy(torch.sum(torch.abs(f - q))) == 0)
                print(
                    f'{model_name} output {i} error: max - {np.max(np.abs(torch_tensor_to_numpy(f) - torch_tensor_to_numpy(q)))}, sum - {np.sum(np.abs(torch_tensor_to_numpy(f) - torch_tensor_to_numpy(q)))}')

                #########################################
                # check model export
                #########################################
                if self.convert_to_fx:
                    # check export to fx
                    # cannot convert to fx when the model has torch.Tensor operations (i.e. tensor.size())
                    fx_model = symbolic_trace(quantized_model)
                # check export to torchscript
                torch_traced = torch.jit.trace(quantized_model, input_x)
                torch_script_model = torch.jit.script(torch_traced)

    def run_test(self, seed=0):
        np.random.seed(seed)
        random.seed(a=seed)
        torch.random.manual_seed(seed)
        input_shapes = self.create_inputs_shape()
        x = self.generate_inputs(input_shapes)

        def representative_data_gen_experimental():
            for _ in range(self.num_calibration_iter):
                yield x

        ptq_models, quantization_info = {}, None
        model_float = self.create_feature_network(input_shapes)
        model_float.eval()
        core_config_dict = self.get_core_configs()
        tpc_dict = self.get_tpc()
        assert isinstance(tpc_dict, dict), "Pytorch tests get_tpc should return a dictionary " \
                                           "mapping the test model name to a TPC object."
        for model_name in tpc_dict.keys():
            tpc = tpc_dict[model_name]
            assert isinstance(tpc, TargetPlatformCapabilities)

            core_config = core_config_dict.get(model_name)
            assert core_config is not None, f"Model name {model_name} does not exists in the test's " \
                                            f"core configs dictionary keys"

            ptq_model, quantization_info = mct.ptq.pytorch_post_training_quantization(in_module=model_float,
                                                                                      representative_data_gen=representative_data_gen_experimental,
                                                                                      target_resource_utilization=self.get_resource_utilization(),
                                                                                      core_config=core_config,
                                                                                      target_platform_capabilities=tpc)

            ptq_models.update({model_name: ptq_model})
        self.compare(ptq_models, model_float, input_x=x, quantization_info=quantization_info)