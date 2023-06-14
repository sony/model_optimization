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


class TestNoQuantKerasFQExporter(KerasFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_tpc(self):
        tp = generate_test_tp_model({'enable_weights_quantization': False,
                                     'enable_activation_quantization': False})
        return generate_keras_tpc(name="test_no_quant", tp_model=tp)

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2D(3,3)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model

    def run_checks(self):
        assert np.all(self.model.layers[1].kernel == self.loaded_model.layers[1].kernel)
        assert np.all(self.model.layers[1].bias == self.loaded_model.layers[1].bias)
        # Check no holder was added
        assert len(self.loaded_model.layers) == 2



