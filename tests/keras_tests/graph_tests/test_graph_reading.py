# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.xception import Xception
from model_compression_toolkit.keras.reader.reader import model_reader


class TestGraphReading(unittest.TestCase):
    def _base_test(self, model_class):
        model = model_class()
        graph = model_reader(model)
        self.assertEqual(len(graph.nodes()), len(model.layers))

    def test_graph_reading(self):
        for model_class in [MobileNet, MobileNetV2, EfficientNetB0, NASNetMobile, ResNet50, Xception]:
            self._base_test(model_class)


if __name__ == '__main__':
    unittest.main()
