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

import model_compression_toolkit as mct
from model_compression_toolkit.verify_packages import FOUND_ONNXRUNTIME, FOUND_ONNX
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.pytorch_tests.model_tests.feature_models.qat_test import dummy_train


class TestExportingQATModelTorchscript(unittest.TestCase):

    def get_model(self):
        return mobilenet_v2(pretrained=True)

    def get_dataset(self):
        yield [to_torch_tensor(np.random.rand(1, 3, 224, 224)).to(get_working_device())]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 2})
        return generate_pytorch_tpc(name="test_conv2d_2bit_fq_weight", tp_model=tp)

    def get_serialization_format(self):
        return mct.exporter.PytorchExportSerializationFormat.TORCHSCRIPT

    def get_quantization_format(self):
        return mct.exporter.QuantizationFormat.FAKELY_QUANT

    def get_tmp_filepath(self):
        return tempfile.mkstemp('.pt')[1]

    def load_exported_model(self, filepath):
        return torch.load(filepath)

    def infer(self, model, images):
        return model(images)

    def export_qat_model(self):
        model = self.get_model()
        images = next(self.get_dataset())

        self.qat_ready, _ = mct.qat.pytorch_quantization_aware_training_init_experimental(model, self.get_dataset,
                                                                                          target_platform_capabilities=self.get_tpc())

        # Dummy train uses LR 0, thus predictions before and after dummy train should be the same
        a = self.qat_ready(images[0])
        x = torch.randn(100, 3, 224, 224)
        y = torch.randn(100, 1000)
        self.qat_ready = dummy_train(self.qat_ready, x, y)
        b = self.qat_ready(images[0])
        self.assertTrue(torch.max(torch.abs(a - b)) == 0, f'QAT ready model was trained using LR 0 thus predictions should be identical but a diff observed {torch.max(a - b)}')

        # Assert qat_ready can be saved and loaded
        _qat_ready_model_tmp_filepath = tempfile.mkstemp('.pt')[1]
        torch.save(self.qat_ready, _qat_ready_model_tmp_filepath)
        self.qat_ready = torch.load(_qat_ready_model_tmp_filepath)

        self.final_model = mct.qat.pytorch_quantization_aware_training_finalize_experimental(self.qat_ready)

        # Assert final_model can be used for inference, can be saved and loaded
        self.final_model(images[0])
        _final_model_tmp_filepath = tempfile.mkstemp('.pt')[1]
        torch.save(self.final_model, _final_model_tmp_filepath)
        self.final_model = torch.load(_final_model_tmp_filepath)
        self.final_model(images[0])

        self.filepath = self.get_tmp_filepath()
        mct.exporter.pytorch_export_model(self.final_model,
                                          self.filepath,
                                          self.get_dataset,
                                          serialization_format=self.get_serialization_format(),
                                          quantization_format=self.get_quantization_format()
                                          )

        self.loaded_model = self.load_exported_model(self.filepath)
        self.infer(self.loaded_model, images[0])

    def test_exported_qat_model(self):
        self.export_qat_model()
        images = next(self.get_dataset())
        a = self.infer(self.final_model, images[0])
        b = self.infer(self.loaded_model, images[0])
        diff = torch.max(torch.abs(a - b))
        assert diff == 0, f'QAT Model before and after export to torchscript should ' \
                          f'predict identical predictions but diff is ' \
                          f'{diff}'

        kernel_diff = torch.max(self.final_model.features_0_0_bn.get_quantized_weights()[
                                    'weight'] - self.loaded_model.features_0_0_bn.weight).item()
        assert kernel_diff == 0, f'Kernels before and after export should be identical but max diff is {kernel_diff}'

        unique_values_per_channel = [len(torch.unique(self.loaded_model.features_0_0_bn.weight[i, :, :, :])) for i in range(32)]
        assert np.max(
            unique_values_per_channel) <= 4, f'In 2bits weights quantization 4 unique values are expected to' \
                                             f' be in each channel but found {unique_values_per_channel} unique ' \
                                             f'values (one per channel)'
