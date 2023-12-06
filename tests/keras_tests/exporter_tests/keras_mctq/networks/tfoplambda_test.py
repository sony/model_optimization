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
import keras
from keras import Input
from packaging import version
import tensorflow as tf

from tests.keras_tests.exporter_tests.keras_mctq.keras_mctq_exporter_base_test import TestKerasMCTQExport

if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.layers import Conv2D, TFOpLambda, Add, DepthwiseConv2D, Dense
else:
    from keras.layers import Conv2D, TFOpLambda, Add, DepthwiseConv2D, Dense
import tensorflow as tf



class TestTFOpLambdaKerasMCTQExporter(TestKerasMCTQExport):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2D(3, 3)(inputs)
        y = Conv2D(3, 3)(inputs)
        z = tf.concat([x, y], axis=0)
        w = tf.concat([x, y], 0)
        x = tf.add(z, w)
        model = keras.Model(inputs=inputs, outputs=x)
        return model
