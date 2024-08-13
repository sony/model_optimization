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

import tempfile
import unittest

import numpy as np
import torch
from torchvision.models.mobilenetv2 import mobilenet_v2

import mct_quantizers
import model_compression_toolkit as mct
from mct_quantizers import QuantizationMethod
from model_compression_toolkit.verify_packages import FOUND_ONNXRUNTIME, FOUND_ONNX
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import DEFAULT_ONNX_OPSET_VERSION

from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.exporter_tests.base_pytorch_onnx_export_test import BasePytorchONNXCustomOpsExportTest
from tests.pytorch_tests.exporter_tests.custom_ops_tests.test_export_pot_onnx_quantizers import OneLayer
from tests.pytorch_tests.model_tests.feature_models.qat_test import dummy_train
import onnx


class TestExportONNXWeightSymmetric2BitsQuantizers(BasePytorchONNXCustomOpsExportTest):
    def __init__(self, onnx_opset_version=DEFAULT_ONNX_OPSET_VERSION):
        super().__init__(onnx_opset_version=onnx_opset_version)

    def get_model(self):
        return OneLayer(torch.nn.Conv2d, in_channels=3, out_channels=4, kernel_size=5)

    def get_tpc(self):
        tp = generate_test_tp_model({'activation_n_bits': 2,
                                     'weights_n_bits': 2,
                                     'weights_quantization_method': QuantizationMethod.SYMMETRIC,
                                     'activation_quantization_method': QuantizationMethod.SYMMETRIC})
        return generate_pytorch_tpc(name="test_conv2d_2bit_fq_weight", tp_model=tp)


    def compare(self, exported_model, wrapped_quantized_model, quantization_info):
        pot_q_nodes = self._get_onnx_node_by_type(exported_model, "ActivationSymmetricQuantizer")
        assert len(pot_q_nodes)==2, f"Expected to find 2 POT quantizers but found {len(pot_q_nodes)}"

        conv_qparams=self._get_onnx_node_attributes(pot_q_nodes[1])
        assert int(wrapped_quantized_model.layer_activation_holder_quantizer.activation_holder_quantizer.signed) == conv_qparams['signed']
        assert wrapped_quantized_model.layer_activation_holder_quantizer.activation_holder_quantizer.threshold_np.astype(np.float32) == conv_qparams['threshold']
        assert wrapped_quantized_model.layer_activation_holder_quantizer.activation_holder_quantizer.num_bits == conv_qparams['num_bits']

        pot_q_nodes = self._get_onnx_node_by_type(exported_model, "WeightsSymmetricQuantizer")
        assert len(pot_q_nodes) == 1, f"Expected to find 1 weight POT quantizer but found {len(pot_q_nodes)}"

        conv_qparams = self._get_onnx_node_attributes(pot_q_nodes[0])
        assert conv_qparams['signed']==1 # Weights always signed
        assert conv_qparams['num_bits'] == wrapped_quantized_model.layer.weights_quantizers['weight'].num_bits
        assert conv_qparams['per_channel'] == int(wrapped_quantized_model.layer.weights_quantizers['weight'].per_channel)
        assert conv_qparams['channel_axis'] == wrapped_quantized_model.layer.weights_quantizers['weight'].channel_axis

        conv_qparams = self._get_onnx_node_const_inputs(exported_model, "WeightsSymmetricQuantizer")
        assert np.all(conv_qparams[0]==wrapped_quantized_model.layer.weights_quantizers['weight'].threshold_np.astype(np.float32))
