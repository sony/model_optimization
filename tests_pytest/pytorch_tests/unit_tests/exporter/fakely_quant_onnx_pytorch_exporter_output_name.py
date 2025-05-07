# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import pytest
import torch
import torch.nn as nn

from model_compression_toolkit.core.pytorch.utils import set_model
from model_compression_toolkit.exporter.model_exporter.pytorch.fakely_quant_onnx_pytorch_exporter import \
    FakelyQuantONNXPyTorchExporter
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import DEFAULT_ONNX_OPSET_VERSION
from model_compression_toolkit.exporter.model_wrapper import is_pytorch_layer_exportable


class SingleOutputModel(nn.Module):
    def __init__(self):
        super(SingleOutputModel, self).__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, x):
        return self.linear(x)


class MultipleOutputModel(nn.Module):
    def __init__(self):
        super(MultipleOutputModel, self).__init__()
        self.linear = nn.Linear(8, 5)

    def forward(self, x):
        return self.linear(x), x, x + 2


class TestONNXExporter:
    test_input_1 = None
    test_expected_1 = ['output']

    test_input_2 = ['output_2']
    test_expected_2 = ['output_2']

    test_input_3 = None
    test_expected_3 = ['output_0', 'output_1', 'output_2']

    test_input_4 = ['out', 'out_11', 'out_22']
    test_expected_4 = ['out', 'out_11', 'out_22']

    test_input_5 = ['out', 'out_11', 'out_22', 'out_33']
    test_expected_5 = ("Mismatch between number of requested output names (['out', 'out_11', 'out_22', 'out_33']) and "
                       "model output count (3):\n")

    def representative_data_gen(self, shape=(3, 8, 8), num_inputs=1, batch_size=2, num_iter=1):
        for _ in range(num_iter):
            yield [torch.randn(batch_size, *shape)] * num_inputs

    def get_exporter(self, model, save_model_path):
        return FakelyQuantONNXPyTorchExporter(model,
                                              is_pytorch_layer_exportable,
                                              save_model_path,
                                              self.representative_data_gen,
                                              onnx_opset_version=DEFAULT_ONNX_OPSET_VERSION)

    def export_model(self, model, save_model_path, output_names, expected_output_names):
        exporter = self.get_exporter(model, save_model_path)

        exporter.export(output_names)

        assert save_model_path.exists(), "ONNX file was not created"
        assert save_model_path.stat().st_size > 0, "ONNX file is empty"

        # Load the ONNX model and check outputs
        onnx_model = onnx.load(str(save_model_path))
        outputs = onnx_model.graph.output

        # Check number of outputs
        assert len(outputs) == len(
            expected_output_names), f"Expected {len(expected_output_names)} output, but found {len(outputs)}"

        found_output_names = [output.name for output in outputs]
        assert found_output_names == expected_output_names, (
            f"Expected output name '{expected_output_names}' found {found_output_names}"
        )

    @pytest.mark.parametrize(
        ("model", "output_names", "expected_output_names"), [
            (SingleOutputModel(), test_input_1, test_expected_1),
            (SingleOutputModel(), test_input_2, test_expected_2),
            (MultipleOutputModel(), test_input_3, test_expected_3),
            (MultipleOutputModel(), test_input_4, test_expected_4),
        ])
    def test_output_model_name(self, tmp_path, model, output_names, expected_output_names):
        save_model_path = tmp_path / "model.onnx"
        set_model(model)

        self.export_model(model, save_model_path, output_names=output_names,
                          expected_output_names=expected_output_names)

    @pytest.mark.parametrize(
        ("model", "output_names", "expected_output_names"), [
            (MultipleOutputModel(), test_input_5, test_expected_5),
        ])
    def test_wrong_number_output_model_name(self, tmp_path, model, output_names, expected_output_names):
        save_model_path = tmp_path / "model.onnx"
        set_model(model)

        try:
            self.export_model(model, save_model_path, output_names=output_names,
                              expected_output_names=expected_output_names)
        except Exception as e:
            assert expected_output_names == str(e)
