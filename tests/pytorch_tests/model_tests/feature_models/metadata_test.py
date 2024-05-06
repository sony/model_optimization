# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
import onnx
import torch
import torch.nn as nn
import numpy as np
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, torch_tensor_to_numpy, set_model
from tests.pytorch_tests.model_tests.base_pytorch_feature_test import BasePytorchFeatureNetworkTest
from tests.common_tests.helpers.tensors_compare import cosine_similarity
from model_compression_toolkit.target_platform_capabilities.constants import IMX500_TP_MODEL
from model_compression_toolkit.constants import PYTORCH
from mct_quantizers import PytorchQuantizationWrapper
from mct_quantizers.pytorch.metadata import add_metadata, get_metadata, add_onnx_metadata, get_onnx_metadata
import tempfile
import os

tp = mct.target_platform


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 13, 1)

    def forward(self, x):
        return self.conv(x)


class MetadataTest(BasePytorchFeatureNetworkTest):

    def get_tpc(self):
        return mct.get_target_platform_capabilities(PYTORCH, IMX500_TP_MODEL, "v2")

    def create_networks(self):
        return DummyNet()

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(get_metadata(quantized_model)) > 0,
                                  msg='A model quantized with TPC IMX500.v2 should have a metadata.')

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            mct.exporter.pytorch_export_model(model=quantized_model,
                                              save_model_path=tmp.name,
                                              repr_dataset=self.representative_data_gen_experimental)

            loaded_onnx_model = onnx.load(tmp.name)
            metadata = get_onnx_metadata(loaded_onnx_model)
            self.unit_test.assertTrue(len(metadata) > 0,
                                      msg='A model quantized with TPC IMX500.v2 should have a metadata.')

