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



class CropAndResizeTest(BaseKerasLayerTest):
    def __init__(self, unit_test):
        super().__init__(unit_test,
                         num_of_inputs=1)

    def get_layers(self):
        boxes = tf.random.uniform(shape=(5, 4))
        box_indices = tf.random.uniform(shape=(5,), minval=0,
                                        maxval=1, dtype=tf.int32)
        return [partial(tf.image.crop_and_resize, boxes=boxes, box_indices=box_indices, crop_size=(24, 24)),
                partial(tf.image.crop_and_resize, boxes=boxes, box_indices=box_indices, crop_size=(24, 24), extrapolation_value=0)]
