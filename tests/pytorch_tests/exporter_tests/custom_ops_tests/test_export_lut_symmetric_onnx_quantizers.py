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

import numpy as np
import torch
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.v1.tpc_pytorch import \
    generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.exporter_tests.base_pytorch_onnx_export_test import BasePytorchONNXCustomOpsExportTest
from tests.pytorch_tests.exporter_tests.custom_ops_tests.test_export_pot_onnx_quantizers import OneLayer


class TestExportONNXWeightLUTSymmetric2BitsQuantizers(BasePytorchONNXCustomOpsExportTest):

    def get_model(self):
        return OneLayer(torch.nn.Conv2d, in_channels=3, out_channels=4, kernel_size=5)

    def get_tpc(self):
        tp = generate_test_tp_model({'activation_n_bits': 2,
                                     'weights_n_bits': 2,
                                     'weights_quantization_method': QuantizationMethod.LUT_SYM_QUANTIZER,
                                     'activation_quantization_method': QuantizationMethod.POWER_OF_TWO})
        return generate_pytorch_tpc(name="test_conv2d_2bit_fq_weight", tp_model=tp)


    def compare(self, exported_model, wrapped_quantized_model, quantization_info, onnx_op_to_search="WeightsLUTSymmetricQuantizer"):
        lut_sym_q_nodes = self._get_onnx_node_by_type(exported_model, onnx_op_to_search)
        assert len(lut_sym_q_nodes)==1, f"Expected to find 1 weight lut symmetric quantizer but found {len(lut_sym_q_nodes)}"

        conv_qparams=self._get_onnx_node_attributes(lut_sym_q_nodes[0])
        assert conv_qparams['signed']==1 # Weights always signed
        assert conv_qparams['num_bits'] == wrapped_quantized_model.layer.weights_quantizers['weight'].num_bits
        assert conv_qparams['per_channel'] == int(wrapped_quantized_model.layer.weights_quantizers['weight'].per_channel)
        assert conv_qparams['channel_axis'] == wrapped_quantized_model.layer.weights_quantizers['weight'].channel_axis
        assert np.isclose(conv_qparams['eps'], wrapped_quantized_model.layer.weights_quantizers['weight'].eps)
        assert conv_qparams['input_rank'] == wrapped_quantized_model.layer.weights_quantizers['weight'].input_rank
        assert conv_qparams['lut_values_bitwidth'] == wrapped_quantized_model.layer.weights_quantizers['weight'].lut_values_bitwidth

        conv_qparams = self._get_onnx_node_const_inputs(exported_model, onnx_op_to_search)
        assert np.all(conv_qparams[0] == wrapped_quantized_model.layer.weights_quantizers['weight'].lut_values)
        assert np.all(conv_qparams[1] == wrapped_quantized_model.layer.weights_quantizers['weight'].threshold)



class TestExportONNXWeightLUTPOT2BitsQuantizers(TestExportONNXWeightLUTSymmetric2BitsQuantizers):

    def get_model(self):
        return OneLayer(torch.nn.Conv2d, in_channels=3, out_channels=4, kernel_size=5)

    def get_tpc(self):
        tp = generate_test_tp_model({'activation_n_bits': 2,
                                     'weights_n_bits': 2,
                                     'weights_quantization_method': QuantizationMethod.LUT_POT_QUANTIZER,
                                     'activation_quantization_method': QuantizationMethod.POWER_OF_TWO})
        return generate_pytorch_tpc(name="test_conv2d_2bit_fq_weight", tp_model=tp)


    def compare(self, exported_model, wrapped_quantized_model, quantization_info):
        super(TestExportONNXWeightLUTPOT2BitsQuantizers, self).compare(exported_model,
                                                                       wrapped_quantized_model,
                                                                       quantization_info,
                                                                       onnx_op_to_search="WeightsLUTPOTQuantizer")

