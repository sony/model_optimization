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


from tests_pytest._fw_tests_common_base.fusing.base_graph_with_fusing_metadata_test import BaseGraphWithFusingMetadataTest
from tests_pytest.keras_tests.keras_test_util.keras_test_mixin import KerasFwMixin

import keras

class TestGraphWithFusionMetadataKeras(BaseGraphWithFusingMetadataTest, KerasFwMixin):

    layer_class_relu = keras.layers.ReLU

    def _data_gen(self):
        return self.get_basic_data_gen(shapes=[(1, 3, 5, 5)])()

    def _get_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(3, (3, 3), activation=None, input_shape=(5, 5, 3), name='conv'),
            keras.layers.ReLU(name='relu'),
            keras.layers.Flatten(name='flatten'),
            keras.layers.Dense(10, name='linear'),
            keras.layers.Softmax(name='softmax')
        ])
        return model
