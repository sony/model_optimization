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
import os
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import Conv2d, ReLU
from torchvision.models.mobilenetv2 import mobilenet_v2

import model_compression_toolkit as mct
from model_compression_toolkit.constants import FOUND_ONNX, FOUND_ONNXRUNTIME, PYTORCH
from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor
from model_compression_toolkit.exporter import pytorch_export_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_pytorch_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit import get_target_platform_capabilities

DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

_, SAVED_MODEL_PATH_PTH = tempfile.mkstemp('.pth')
_, SAVED_MODEL_PATH_ONNX = tempfile.mkstemp('.onnx')
_, SAVED_MP_MODEL_PATH_ONNX = tempfile.mkstemp('.onnx')


class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv1 = Conv2d(1, 3, 8, bias=True)
        self.relu = ReLU()

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu(x)
        return x


if FOUND_ONNX:
    import onnx


    class TestPyTorchFakeQuantExporter(unittest.TestCase):

        def tearDown(self):
            os.remove(SAVED_MODEL_PATH_PTH)
            os.remove(SAVED_MODEL_PATH_ONNX)

        def repr_datagen(self):
            for _ in range(1):
                yield [np.random.random((1, 3, 224, 224))]

        def setUp(self) -> None:
            self.model = mobilenet_v2(pretrained=True)
            self.exportable_model = self.run_mct(self.model, new_experimental_exporter=True)
            self.exportable_model.eval()
            pytorch_export_model(model=self.exportable_model,
                                 save_model_path=SAVED_MODEL_PATH_PTH,
                                 repr_dataset=self.repr_datagen,
                                 target_platform_capabilities=self.tpc,
                                 serialization_format=mct.exporter.PytorchExportSerializationFormat.TORCHSCRIPT)
            self.exported_model_pth = torch.load(SAVED_MODEL_PATH_PTH)
            self.exported_model_pth.eval()
            pytorch_export_model(model=self.exportable_model,
                                 save_model_path=SAVED_MODEL_PATH_ONNX,
                                 repr_dataset=self.repr_datagen,
                                 target_platform_capabilities=self.tpc,
                                 serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX)
            self.exported_model_onnx = onnx.load(SAVED_MODEL_PATH_ONNX)
            # Check that the model is well formed
            onnx.checker.check_model(self.exported_model_onnx)

        def run_mct(self, model, new_experimental_exporter):
            core_config = mct.core.CoreConfig()
            self.tpc = self.get_tpc()
            new_export_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(
                in_module=model,
                core_config=core_config,
                representative_data_gen=self.repr_datagen,
                target_platform_capabilities=self.tpc,
                new_experimental_exporter=new_experimental_exporter)
            return new_export_model

        def get_tpc(self):
            return DEFAULT_PYTORCH_TPC

        def test_pth_predictions(self):
            images = to_torch_tensor(next(self.repr_datagen()))[0]
            diff = self.exported_model_pth(images) - self.exportable_model(images)
            max_abs_error = np.max(np.abs(diff.cpu().detach().numpy()))
            print(f'Max abs error: {max_abs_error}')
            self.assertTrue(max_abs_error == 0)

        if FOUND_ONNXRUNTIME:
            def test_onnx_inference(self):
                import onnxruntime

                ort_session = onnxruntime.InferenceSession(SAVED_MODEL_PATH_ONNX)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

                x = to_torch_tensor(next(self.repr_datagen()))[0]
                # compute ONNX Runtime output prediction
                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
                ort_session.run(None, ort_inputs)


    class TestPyTorch2BitONNXExporter(unittest.TestCase):

        def tearDown(self):
            os.remove(SAVED_MP_MODEL_PATH_ONNX)

        def repr_datagen(self):
            for _ in range(1):
                yield [np.random.random((1, 1, 224, 224))]

        def setUp(self) -> None:
            self.model = BasicModel()
            self.exportable_model = self.run_mct(self.model, new_experimental_exporter=True)
            self.exportable_model.eval()
            pytorch_export_model(model=self.exportable_model,
                                 save_model_path=SAVED_MP_MODEL_PATH_ONNX,
                                 repr_dataset=self.repr_datagen,
                                 target_platform_capabilities=self.get_tpc(),
                                 serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX)
            self.exported_model_onnx = onnx.load(SAVED_MP_MODEL_PATH_ONNX)
            # Check that the model is well formed
            onnx.checker.check_model(self.exported_model_onnx)

        def get_tpc(self):
            return generate_pytorch_tpc(name="2_quant_pytorch_test",
                                        tp_model=generate_test_tp_model({'weights_n_bits': 2,
                                                                         'activation_n_bits': 8,
                                                                         'enable_weights_quantization': True,
                                                                         'enable_activation_quantization': True
                                                                         }))

        def run_mct(self, model, new_experimental_exporter):
            core_config = mct.core.CoreConfig()
            new_export_model, _ = mct.ptq.pytorch_post_training_quantization_experimental(
                in_module=model,
                core_config=core_config,
                representative_data_gen=self.repr_datagen,
                target_platform_capabilities=self.get_tpc(),
                new_experimental_exporter=new_experimental_exporter)
            return new_export_model

        if FOUND_ONNXRUNTIME:
            def test_onnx_inference(self):
                import onnxruntime

                ort_session = onnxruntime.InferenceSession(SAVED_MP_MODEL_PATH_ONNX)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

                # get onnx conv weight
                conv_weight_name = 'conv1.weight'
                from onnx import numpy_helper
                INTIALIZERS = self.exported_model_onnx.graph.initializer
                Weight = []
                for initializer in INTIALIZERS:
                    if initializer.name == conv_weight_name:
                        W = numpy_helper.to_array(initializer)
                        Weight.append(W)

                onnx_unique_values = np.unique(Weight[0])
                exportable_model_unique_values = torch.unique(self.exportable_model.conv1.get_quantized_weights()
                                                              ['weight']).detach().cpu().numpy()
                self.assertTrue(np.all(onnx_unique_values == exportable_model_unique_values))

                x = to_torch_tensor(next(self.repr_datagen()))[0]
                # compute ONNX Runtime output prediction
                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
                onnx_output = ort_session.run(None, ort_inputs)

                exportable_model_output = self.exportable_model(x)
                diff = exportable_model_output.detach().cpu().numpy() - onnx_output[0]
                max_abs_error = np.max(np.abs(diff))
                print(f'Max abs error: {max_abs_error}')
                self.assertTrue(max_abs_error == 0)
