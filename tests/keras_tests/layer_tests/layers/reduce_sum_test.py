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
from functools import partial

import tensorflow as tf
from tests.keras_tests.layer_tests.base_keras_layer_test import BaseKerasLayerTest



class ReduceSumTest(BaseKerasLayerTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_layers(self):
        return [partial(tf.reduce_sum, axis=0),
                partial(tf.reduce_sum, axis=1),
                partial(tf.reduce_sum, axis=0, keepdims=True),
                partial(tf.reduce_sum, axis=0, keepdims=False)]
