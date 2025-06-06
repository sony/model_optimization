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
    from keras.src.layers import Conv2D, TFOpLambda, Add, DepthwiseConv2D, Dense, Conv2DTranspose
else:
    from keras.layers import Conv2D, TFOpLambda, Add, DepthwiseConv2D, Dense, Conv2DTranspose
import numpy as np
import tensorflow as tf

from tests.common_tests.helpers.generate_test_tpc import generate_test_tpc
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.keras_tests.exporter_tests.keras_fake_quant.keras_fake_quant_exporter_base_test import \
    get_minmax_from_qparams


class TestConv2DTransposeKerasMCTQExporter(TestKerasMCTQExport):

    def get_input_shape(self):
        return [(30, 30, 3)]

    def get_tpc(self):
        tp = generate_test_tpc({'weights_n_bits': 2,
                                     'activation_n_bits': 2})
        return generate_keras_tpc(name="test_conv2d_2bit_fq_weight", tpc=tp)

    def get_model(self):
        inputs = Input(shape=self.get_input_shape()[0])
        x = Conv2DTranspose(6, 20)(inputs)
        model = keras.Model(inputs=inputs, outputs=x)
        return model
