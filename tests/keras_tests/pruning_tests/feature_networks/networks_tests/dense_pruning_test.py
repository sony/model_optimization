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


import tensorflow as tf

import model_compression_toolkit as mct
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import generate_keras_tpc
from tests.common_tests.helpers.generate_test_tp_model import generate_test_tp_model
from tests.common_tests.pruninig.constant_importance_metric import add_const_importance_metric, \
    ConstImportanceMetric

from tests.keras_tests.pruning_tests.feature_networks.pruning_keras_feature_test import PruningKerasFeatureTest
from tests.keras_tests.utils import get_layers_from_model_by_type
import numpy as np

keras = tf.keras
layers = keras.layers


class DensePruningTest(PruningKerasFeatureTest):
    """
    Test a network with two adjacent dense and check it's pruned for a target compression ratio.
    """

    def __init__(self,
                 unit_test,
                 use_bn=False,
                 activation_layer=None,
                 simd=1,
                 use_constant_importance_metric=True):

        super().__init__(unit_test,
                         input_shape=(8, 8, 3))
        self.use_bn = use_bn
        self.activation_layer = activation_layer
        self.simd = simd
        self.use_constant_importance_metric = use_constant_importance_metric

    def get_tpc(self):
        tp = generate_test_tp_model({'simd_size': self.simd})
        return generate_keras_tpc(name="simd_test", tp_model=tp)

    def get_pruning_config(self):
        if self.use_constant_importance_metric:
            add_const_importance_metric(first_num_oc=10, second_num_oc=6, simd=self.simd)
            return mct.pruning.PruningConfig(importance_metric=ConstImportanceMetric.CONST)
        return super().get_pruning_config()

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Dense(units=10)(inputs)
        if self.use_bn:
            x = layers.BatchNormalization()(x)
        if self.activation_layer:
            x = self.activation_layer(x)
        x = layers.Dense(units=6)(x)
        outputs = layers.Dense(units=6)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def get_kpi(self):
        # Remove only one group of channels only one parameter should be pruned
        return mct.KPI(weights_memory=(self.dense_model_num_params - 1) * 4)

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        dense_layers = get_layers_from_model_by_type(float_model, layers.Dense)
        prunable_layers = get_layers_from_model_by_type(quantized_model, layers.Dense)

        is_first_layer_pruned = prunable_layers[0].units == 10 - self.simd
        is_second_layer_pruned = prunable_layers[1].units == 6 - self.simd

        # Make sure only one of layers has been pruned
        self.unit_test.assertTrue(is_first_layer_pruned != is_second_layer_pruned)

        # In constant case, the last SIMD channels of the first layer should be pruned:
        if self.use_constant_importance_metric:
            self.unit_test.assertTrue(is_first_layer_pruned)
            self.unit_test.assertTrue(np.all(prunable_layers[0].kernel.numpy() == dense_layers[0].kernel.numpy()[:, :-self.simd]))
            self.unit_test.assertTrue(np.all(prunable_layers[0].bias.numpy() == dense_layers[0].bias.numpy()[:-self.simd]))

            # Make sure the only in channel removed is the last channel of the second dense layer
            self.unit_test.assertTrue(np.all(prunable_layers[1].kernel.numpy() == dense_layers[1].kernel.numpy()[:-self.simd, :]))
            self.unit_test.assertTrue(np.all(prunable_layers[1].bias.numpy() == dense_layers[1].bias.numpy()))

        if is_first_layer_pruned:
            self.unit_test.assertTrue(np.all(prunable_layers[2].kernel.numpy() == dense_layers[2].kernel.numpy()))
            self.unit_test.assertTrue(np.all(prunable_layers[2].bias.numpy() == dense_layers[2].bias.numpy()))
