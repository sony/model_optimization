#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================
from tests.pytorch_tests.exporter_tests.base_pytorch_onnx_export_test import BasePytorchONNXExportTest
from torch import nn

class TestExportONNXMultipleInputs(BasePytorchONNXExportTest):
    def get_model(self):
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, input1, input2):
                x1 = self.conv1(input1)
                x2 = self.conv2(input2)
                return x1 + x2

        return Model()

    def get_input_shapes(self):
        return [(1, 3, 8, 8), (1, 3, 8, 8)]

    def compare(self, loaded_model, quantized_model, quantization_info):
        assert len(loaded_model.graph.input)==2, f"Model expected to have two inputs but has {len(loaded_model.graph.input)}"
        self.infer(loaded_model, next(self.get_dataset()))