# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
import model_compression_toolkit as mct
from tests.keras_tests.feature_networks_tests.base_keras_feature_test import BaseKerasFeatureNetworkTest
if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Activation, ReLU, Add
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Activation, ReLU, Add

keras = tf.keras
layers = keras.layers
activations = keras.activations
tp = mct.target_platform


class BaseLayerFusingTest(BaseKerasFeatureNetworkTest):

    def __init__(self, unit_test):
        super(BaseLayerFusingTest, self).__init__(unit_test=unit_test, input_shape=(16, 16, 3))
        self.expected_fusions = []

    def get_type(self, fusion):
        fusion_types = [x.type for x in fusion]
        return fusion_types

    def get_tpc(self):
        default_config, mixed_precision_cfg_list = get_op_quantization_configs()
        default_configuration_options = tp.QuantizationConfigOptions([default_config])
        generated_tp = tp.TargetPlatformModel(default_configuration_options, name='layer_fusing_test')
        mixed_precision_configuration_options = tp.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                             base_config=default_config)
        return generated_tp, mixed_precision_configuration_options

    def compare(self, quantized_model, float_model, input_x=None, quantization_info=None):
        self.unit_test.assertTrue(len(quantization_info.fusions) == len(self.expected_fusions), msg=f'Number of fusions is not as expected!')
        for i,fusion in enumerate(quantization_info.fusions):
            self.unit_test.assertTrue(self.get_type(fusion) == self.expected_fusions[i], msg=f'Miss-match fusion compared to expected!')


class LayerFusingTest1(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2D, Activation]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            any_relu = tp.OperatorsSet("AnyReLU")
            # Define fusions
            tp.Fusing([conv, any_relu])

        keras_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with keras_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2D])
            tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                                 tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                 tp.LayerFilterParams(Activation, activation="relu")])
        return keras_tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=16, kernel_size=(3, 3))(inputs)
        y = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)


class LayerFusingTest2(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2D, Activation], [Conv2D, ReLU], [Conv2D, tf.nn.sigmoid], [Conv2D, Activation]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            any_relu = tp.OperatorsSet("AnyReLU")
            swish = tp.OperatorsSet("Swish")
            sigmoid = tp.OperatorsSet("Sigmoid")
            tanh = tp.OperatorsSet("Tanh")
            activations_after_conv_to_fuse = tp.OperatorSetConcat(any_relu, swish, sigmoid, tanh)
            # Define fusions
            tp.Fusing([conv, activations_after_conv_to_fuse])

        keras_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with keras_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2D, DepthwiseConv2D])
            tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                                 tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                 tp.LayerFilterParams(Activation, activation="relu")])
            tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])
            tp.OperationsSetToLayers("Sigmoid", [tf.nn.sigmoid, tp.LayerFilterParams(Activation, activation="sigmoid")])
            tp.OperationsSetToLayers("Tanh", [tf.nn.tanh, tp.LayerFilterParams(Activation, activation="tanh")])
        return keras_tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=16, kernel_size=(3, 3))(inputs)
        x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='tanh')(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
        x = activations.sigmoid(x)
        y = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='swish')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)


class LayerFusingTest3(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2D, Activation]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            any_relu = tp.OperatorsSet("AnyReLU")
            # Define fusions
            tp.Fusing([conv, any_relu])

        keras_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with keras_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2D])
            tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                                 tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                 tp.LayerFilterParams(Activation, activation="relu")])
        return keras_tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=16, kernel_size=(3, 3))(inputs)
        x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='tanh')(x)
        x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
        x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
        x = activations.sigmoid(x)
        y = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='swish')(x)
        return tf.keras.models.Model(inputs=inputs, outputs=y)


class LayerFusingTest4(BaseLayerFusingTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)
        self.expected_fusions = [[Conv2D, Activation, Add], [Conv2D, Activation, Add], [Conv2D, Activation], [Conv2D, ReLU, Add], [Dense, tf.nn.silu], [Dense, Activation]]

    def get_tpc(self):
        generated_tp, mixed_precision_configuration_options = super().get_tpc()
        with generated_tp:
            conv = tp.OperatorsSet("Conv", mixed_precision_configuration_options)
            fc = tp.OperatorsSet("FullyConnected", mixed_precision_configuration_options)
            any_relu = tp.OperatorsSet("AnyReLU")
            add = tp.OperatorsSet("Add")
            swish = tp.OperatorsSet("Swish")
            activations_to_fuse = tp.OperatorSetConcat(any_relu, swish)
            # Define fusions
            tp.Fusing([conv, activations_to_fuse])
            tp.Fusing([conv, add, activations_to_fuse])
            tp.Fusing([conv, activations_to_fuse, add])
            tp.Fusing([fc, activations_to_fuse])

        keras_tpc = tp.TargetPlatformCapabilities(generated_tp, name='layer_fusing_test')
        with keras_tpc:
            tp.OperationsSetToLayers("Conv", [Conv2D])
            tp.OperationsSetToLayers("FullyConnected", [Dense])
            tp.OperationsSetToLayers("AnyReLU", [tf.nn.relu,
                                                 tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                 tp.LayerFilterParams(Activation, activation="relu")])
            tp.OperationsSetToLayers("Add", [tf.add, Add])
            tp.OperationsSetToLayers("Swish", [tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")])

        return keras_tpc

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='swish')(inputs)
        x1 = layers.Add()([x, inputs])
        x2 = layers.Conv2D(filters=3, kernel_size=(2, 2), padding='same', activation='swish')(x1)
        x2 = layers.Add()([x1, x2])
        x2 = layers.Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(x2)
        x3 = layers.Conv2D(filters=3, kernel_size=(2, 2), padding='same')(x2)
        x3 = layers.ReLU()(x3)
        x3 = layers.Add()([x2, x3])
        x3 = layers.Flatten()(x3)
        x3 = layers.Dense(units=16)(x3)
        x3 = activations.swish(x3)
        y = layers.Dense(units=16, activation='swish')(x3)
        return tf.keras.models.Model(inputs=inputs, outputs=y)
