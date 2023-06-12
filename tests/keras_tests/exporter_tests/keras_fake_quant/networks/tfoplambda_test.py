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
from keras.applications import MobileNetV2
from keras.layers import Conv2D, TFOpLambda, Add, DepthwiseConv2D, Dense
import numpy as np
import tensorflow as tf

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.default_tpc.latest import generate_keras_tpc
from tests.keras_tests.exporter_tests.keras_fake_quant.keras_fake_quant_exporter_base_test import \
    KerasFakeQuantExporterBaseTest


class TestTFOpLambdaKerasFQExporter(KerasFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2D(3,3)(inputs)
        y = Conv2D(3,3)(inputs)
        z = tf.concat([x,y],axis=0)
        w = tf.concat([x, y], 0)
        x = tf.add(z,w)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_checks(self):
        assert len(self.loaded_model.layers)==12
        assert isinstance(self.loaded_model.layers[1], TFOpLambda)
        assert self.loaded_model.layers[1].function==tf.quantization.fake_quant_with_min_max_vars

        assert isinstance(self.loaded_model.layers[6], TFOpLambda)
        assert self.loaded_model.layers[6].function == tf.concat
        assert isinstance(self.loaded_model.layers[8], TFOpLambda)
        assert self.loaded_model.layers[8].function == tf.quantization.fake_quant_with_min_max_vars
        assert self.loaded_model.layers[6].output.ref() == self.loaded_model.layers[8].input.ref()

        assert isinstance(self.loaded_model.layers[7], TFOpLambda)
        assert self.loaded_model.layers[7].function == tf.concat
        assert isinstance(self.loaded_model.layers[9], TFOpLambda)
        assert self.loaded_model.layers[9].function == tf.quantization.fake_quant_with_min_max_vars
        assert self.loaded_model.layers[7].output.ref() == self.loaded_model.layers[9].input.ref()

        assert isinstance(self.loaded_model.layers[10], TFOpLambda)
        assert self.loaded_model.layers[10].function == tf.add
        assert self.loaded_model.layers[10].input.ref() == self.loaded_model.layers[8].output.ref()
        assert self.loaded_model.layers[10].inbound_nodes[0].call_kwargs['y'].ref() == self.loaded_model.layers[9].output.ref()




