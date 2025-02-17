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


import unittest
import torch
import numpy as np
from model_compression_toolkit.core.pytorch import pytorch_implementation

from model_compression_toolkit.core.pytorch.reader.reader import fx_graph_module_generation
from model_compression_toolkit.core.pytorch.pytorch_implementation import to_torch_tensor, PytorchImplementation


class BadFxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs, flag=False):
        x = self.conv(inputs)
        if flag:
            x = self.relu(x)
        else:
            x = self.relu(x) + x
        return x


def data_gen():
    yield [np.zeros((1, 3, 20, 20))]


class TestGraphReading(unittest.TestCase):
    def test_fx_tracer_error(self):
        model = BadFxModel()

        with self.assertRaises(Exception) as e:
            fx_graph_module_generation(model,
                                       data_gen,
                                       to_torch_tensor)
        self.assertEqual(str(e.exception).split('\n')[0], 'Error parsing model with torch.fx')

    def test_disconnected_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x, unused=None):
                return self.conv(x)

        with self.assertRaises(ValueError) as e:
            PytorchImplementation().model_reader(Model(), data_gen)
        self.assertEqual(str(e.exception), r"The network contains disconnected input(s): ['unused'].")


if __name__ == '__main__':
    unittest.main()
