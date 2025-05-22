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

from mct_quantizers.pytorch.metadata import add_metadata, get_onnx_metadata
from model_compression_toolkit.core import QuantizationConfig
from model_compression_toolkit.core.pytorch.back2framework.float_model_builder import FloatPyTorchModel
from model_compression_toolkit.exporter.model_exporter.pytorch.pytorch_export_facade import pytorch_export_model
from tests_pytest.pytorch_tests.torch_test_util.torch_test_mixin import BaseTorchIntegrationTest


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 8)

    def forward(self, x):
        x = self.linear(x)
        return self.linear2(x)


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class TestONNXExporterMetadata(BaseTorchIntegrationTest):
    # Module names, metadatas
    test_input_1 = [[None], [{'author': 'John Doe', 'model version': '1'}]]
    test_expected_1 = [{'author': 'John Doe', 'model version': '1'}]

    test_input_2 = [["model"], [{'author': 'John Doe', 'model version': '1'}]]
    test_expected_2 = [{'author': 'John Doe', 'model version': '1'}]

    test_input_3 = [[None, "model"], [{'author': 'John Doe', 'model version': '1'},
                                      {'author': 'John Doe', 'model version': '2'}]]
    test_expected_3 = [{'author': 'John Doe', 'model version': '1'}]

    test_input_4 = [[None], [None]]
    test_expected_4 = [None]

    def representative_data_gen(self, num_inputs=1):
        batch_size, num_iter, shape = 2, 1, (3, 8, 8)

        def data_gen():
            for _ in range(num_iter):
                yield [torch.randn(batch_size, *shape)] * num_inputs

        return data_gen

    def get_pytorch_model(self, model, data_generator, minimal_tpc):
        qc = QuantizationConfig()
        graph = self.run_graph_preparation(model=model, datagen=data_generator, tpc=minimal_tpc,
                                           quant_config=qc)
        pytorch_model = FloatPyTorchModel(graph=graph)
        return pytorch_model

    def add_metadata_to_model(self, model, metadata):
        exportable_model = add_metadata(model, metadata)
        return exportable_model

    def export_model(self, model, save_model_path, data_generator):
        pytorch_export_model(model, save_model_path, data_generator)

        assert save_model_path.exists(), "ONNX file was not created"
        assert save_model_path.stat().st_size > 0, "ONNX file is empty"

        onnx_model = onnx.load(str(save_model_path))
        return onnx_model

    def validate_metadata(self, onnx_model, metadatas, expected_metadatas, caplog):
        messages = [record.getMessage() for record in caplog.records]

        # Check for multiple metadatas warning
        if len(metadatas) > 1:
            assert (f"Attribute 'metadata' found in {len(metadatas)} places. Only the first one was "
                    f"assigned to 'model.metadata'.") == messages[0]

        onnx_metadata = get_onnx_metadata(onnx_model)

        # Check metadata
        if expected_metadatas[0] is not None:
            for key, val in expected_metadatas[0].items():
                assert val == onnx_metadata[key]
        else:
            # No metadata
            assert onnx_metadata == {}
            assert "Attribute 'metadata' not found in the model or its submodules." == messages[0]

    @pytest.mark.parametrize(
        ("model", "modules_names", "metadatas", "expected_metadatas"), [
            (BaseModel(), test_input_1[0], test_input_1[1], test_expected_1),
            (BaseModel(), test_input_2[0], test_input_2[1], test_expected_2),
            (BaseModel(), test_input_3[0], test_input_3[1], test_expected_3),
            (BaseModel(), test_input_4[0], test_input_4[1], test_expected_4),
        ])
    def test_onnx_metadata(self, caplog, tmp_path, model, modules_names, metadatas, expected_metadatas,
                           minimal_tpc):
        save_model_path = tmp_path / "model.onnx"
        data_generator = self.representative_data_gen(num_inputs=1)
        pytorch_model = self.get_pytorch_model(model, data_generator, minimal_tpc)

        pytorch_model = WrappedModel(pytorch_model)

        # Add Metadata
        for module, metadata in zip(modules_names, metadatas):
            if metadata is not None:
                target = getattr(pytorch_model, module) if module is not None else pytorch_model
                self.add_metadata_to_model(target, metadata)

        import logging
        with caplog.at_level(logging.WARNING):
            onnx_model = self.export_model(pytorch_model, save_model_path, data_generator)
        self.validate_metadata(onnx_model, metadatas, expected_metadatas, caplog)
