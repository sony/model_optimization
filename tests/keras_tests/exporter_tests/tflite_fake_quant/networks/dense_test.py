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
from keras.layers import Conv2D, Dense, Flatten
import numpy as np
import tests.keras_tests.exporter_tests.constants as constants


from tests.keras_tests.exporter_tests.tflite_fake_quant.tflite_fake_quant_exporter_base_test import \
    TFLiteFakeQuantExporterBaseTest

from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc


class TestDenseReusedTFLiteFQExporter(TFLiteFakeQuantExporterBaseTest):

    def get_input_shape(self):
        return [(3, 3, 3)]

    def get_tpc(self):
        tp = generate_test_tp_model({'weights_n_bits': 2})
        return generate_keras_tpc(name="test_dense_2bit_reused_weight", tp_model=tp)

    def get_model(self):
        dense = Dense(27)
        inputs = Input(shape=self.get_input_shape()[0])
        x = Flatten()(inputs)
        x = dense(x)
        x = dense(x)
        return keras.Model(inputs=inputs, outputs=x)

    def run_checks(self):
        ops = self.interpreter._get_ops_details()
        op_inputs = []
        for op in ops:
            if op['op_name'] == 'FULLY_CONNECTED':
                op_inputs.append(op['inputs'])
        assert len(op_inputs) == 2, f'Expected to find 2 ops of CONV_2D but found {len(op_inputs)}'
        # Inputs like kernel and bias are expected to be the same since the conv is reused.
        first_inputs = op_inputs[0][1:] # Ignore first input of activations
        second_inputs = op_inputs[1][1:] # Ignore first input of activations
        assert np.all(first_inputs==second_inputs), f'Since conv is reused, the input weight tensors of the op are expected to be identical in both occurrences of the op but indices are {first_inputs} and {second_inputs}'
