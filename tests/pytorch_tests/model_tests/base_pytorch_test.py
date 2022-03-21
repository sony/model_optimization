# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

from model_compression_toolkit import MixedPrecisionQuantizationConfig
from model_compression_toolkit.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
from model_compression_toolkit.pytorch.utils import get_working_device, set_model, to_torch_tensor, \
    torch_tensor_to_numpy
import model_compression_toolkit as mct
import torch
import numpy as np
from tests.common_tests.base_feature_test import BaseFeatureNetworkTest

"""
The base test class for the feature networks
"""
class BasePytorchTest(BaseFeatureNetworkTest):
    def __init__(self, unit_test, float_reconstruction_error=1e-7):
        super().__init__(unit_test)
        self.float_reconstruction_error = float_reconstruction_error

    def get_quantization_configs(self):
        return {
            'no_quantization': mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                                      mct.QuantizationErrorMethod.NOCLIPPING,
                                                      32, 32, False, True, True,
                                                      enable_weights_quantization=False,
                                                      enable_activation_quantization=False),
            'all_32bit': mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                                mct.QuantizationErrorMethod.NOCLIPPING,
                                                32, 32, False, True, True,
                                                enable_weights_quantization=True,
                                                enable_activation_quantization=True),
            'all_4bit': mct.QuantizationConfig(mct.QuantizationErrorMethod.NOCLIPPING,
                                               mct.QuantizationErrorMethod.NOCLIPPING,
                                               4, 4, False, False, True,
                                               enable_weights_quantization=True,
                                               enable_activation_quantization=True),
        }

    def create_inputs_shape(self):
        return [[self.val_batch_size, 3, 32, 32]]

    @staticmethod
    def generate_inputs(input_shapes):
        return to_torch_tensor([torch.randn(*in_shape) for in_shape in input_shapes])

    def create_feature_network(self, input_shape):
        pass

    def compare(self, quantized_models, float_model, input_x=None, quantization_info=None, convert_fx=True):
        set_model(float_model)
        float_result = float_model(*input_x)
        for model_name, quantized_model in quantized_models.items():
            set_model(quantized_model)
            quant_result = quantized_model(*input_x)
            for i, (f, q) in enumerate(zip(float_result, quant_result)):
                if model_name == 'no_quantization':
                    # Check if we have a BatchNorm layer in the model. If so, the outputs will not be the same,
                    # since the sqrt function in the BatchNorm folding is not exactly like the sqrt in the C
                    # implementation of BatchNorm.
                    if torch.nn.BatchNorm2d in [type(module) for name, module in float_model.named_modules()]:
                        self.unit_test.assertTrue(np.all(np.isclose(torch_tensor_to_numpy(f), torch_tensor_to_numpy(q),
                                                                    atol=self.float_reconstruction_error)))
                    else:
                        self.unit_test.assertTrue(torch_tensor_to_numpy(torch.sum(torch.abs(f - q))) == 0)
                elif model_name == 'all_4bit':
                    self.unit_test.assertFalse(torch_tensor_to_numpy(torch.sum(torch.abs(f - q))) == 0)
                print(
                    f'{model_name} output {i} error: max - {np.max(np.abs(torch_tensor_to_numpy(f) - torch_tensor_to_numpy(q)))}, sum - {np.sum(np.abs(torch_tensor_to_numpy(f) - torch_tensor_to_numpy(q)))}')

                #########################################
                # check model export
                #########################################
                if convert_fx:
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

        def representative_data_gen():
            return x

        ptq_models = {}
        model_float = self.create_feature_network(input_shapes)
        for model_name, quant_config in self.get_quantization_configs().items():
            if isinstance(quant_config, MixedPrecisionQuantizationConfig):
                ptq_model, quantization_info = mct.pytorch_post_training_quantization_mixed_precision(model_float,
                                                                                                      representative_data_gen,
                                                                                                      n_iter=self.num_calibration_iter,
                                                                                                      quant_config=quant_config,
                                                                                                      fw_info=DEFAULT_PYTORCH_INFO,
                                                                                                      network_editor=self.get_network_editor(),
                                                                                                      gptq_config=self.get_gptq_config(),
                                                                                                      target_kpi=self.get_kpi())
                ptq_models.update({model_name: ptq_model})
            else:
                ptq_model, quantization_info = mct.pytorch_post_training_quantization(model_float,
                                                                                      representative_data_gen,
                                                                                      n_iter=1,
                                                                                      quant_config=quant_config,
                                                                                      fw_info=DEFAULT_PYTORCH_INFO,
                                                                                      network_editor=self.get_network_editor())
                ptq_models.update({model_name: ptq_model})
        self.compare(ptq_models, model_float, input_x=x, quantization_info=quantization_info)
