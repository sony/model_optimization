import unittest
import numpy as np
import tensorflow as tf

import model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema as schema
from model_compression_toolkit.core import DEFAULTCONFIG, QuantizationConfig
from model_compression_toolkit.core.common.fusion.layer_fusing import fusion
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2fw import \
    CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_models.imx500_tpc.latest import \
    get_op_quantization_configs
import model_compression_toolkit as mct
from tests.common_tests.helpers.prep_graph_for_func_test import prepare_graph_with_configs

if tf.__version__ < "2.6":
    from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, Activation, ReLU, Add
else:
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Activation, ReLU, Add

keras = tf.keras
layers = keras.layers
activations = keras.activations
tp = mct.target_platform

INPUT_SHAPE = (16, 16, 3)


def representative_dataset():
    yield [np.random.randn(1, 16, 16, 3).astype(np.float32)]


def create_network_1(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3))(inputs)
    y = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=y)


def create_network_2(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3))(inputs)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='tanh')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = activations.sigmoid(x)
    y = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='swish')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=y)


def create_network_3(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3))(inputs)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='tanh')(x)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Conv2D(filters=16, kernel_size=(1, 1))(x)
    x = activations.sigmoid(x)
    y = layers.Conv2D(filters=16, kernel_size=(1, 1), activation='swish')(x)
    return tf.keras.models.Model(inputs=inputs, outputs=y)


def create_network_4(input_shape):
    inputs = layers.Input(shape=input_shape)
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


def generate_base_tpc(operator_set, fusing_patterns):
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    default_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(
        [default_config]))
    generated_tp = schema.TargetPlatformModel(
        default_qco=default_configuration_options,
        tpc_minor_version=None,
        tpc_patch_version=None,
        tpc_platform_type=None,
        operator_set=tuple(operator_set),
        fusing_patterns=tuple(fusing_patterns),
        add_metadata=False, name='layer_fusing_test')

    return generated_tp


def get_tpc_1():
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                             base_config=base_config)
    conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
    any_relu = schema.OperatorsSet(name="AnyReLU")
    operator_set = [conv, any_relu]
    # Define fusions
    fusing_patterns = [schema.Fusing(operator_groups=(conv, any_relu))]

    generated_tp = generate_base_tpc(operator_set, fusing_patterns)

    return generated_tp

def get_tpc_2():
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                             base_config=base_config)
    conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
    any_relu = schema.OperatorsSet(name="AnyReLU")
    swish = schema.OperatorsSet(name="Swish")
    sigmoid = schema.OperatorsSet(name="Sigmoid")
    tanh = schema.OperatorsSet(name="Tanh")
    operator_set = [conv, any_relu, swish, sigmoid, tanh]
    activations_after_conv_to_fuse = schema.OperatorSetConcat(operators_set=[any_relu, swish, sigmoid, tanh])
    # Define fusions
    fusing_patterns = [schema.Fusing(operator_groups=(conv, activations_after_conv_to_fuse))]

    generated_tp = generate_base_tpc(operator_set, fusing_patterns)

    return generated_tp

def get_tpc_3():
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                             base_config=base_config)
    conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
    any_relu = schema.OperatorsSet(name="AnyReLU")
    operator_set = [conv, any_relu]
    # Define fusions
    fusing_patterns = [schema.Fusing(operator_groups=(conv, any_relu))]

    generated_tp = generate_base_tpc(operator_set, fusing_patterns)

    return generated_tp


def get_tpc_4():
    base_config, mixed_precision_cfg_list, default_config = get_op_quantization_configs()
    mixed_precision_configuration_options = schema.QuantizationConfigOptions(quantization_configurations=tuple(mixed_precision_cfg_list),
                                                                             base_config=base_config)
    conv = schema.OperatorsSet(name="Conv", qc_options=mixed_precision_configuration_options)
    fc = schema.OperatorsSet(name="FullyConnected", qc_options=mixed_precision_configuration_options)
    any_relu = schema.OperatorsSet(name="AnyReLU")
    add = schema.OperatorsSet(name="Add")
    swish = schema.OperatorsSet(name="Swish")
    activations_to_fuse = schema.OperatorSetConcat(operators_set=[any_relu, swish])
    operator_set = [conv, fc, any_relu, add, swish]
    # Define fusions
    fusing_patterns = [schema.Fusing(operator_groups=(conv, activations_to_fuse)),
                       schema.Fusing(operator_groups=(conv, add, activations_to_fuse)),
                       schema.Fusing(operator_groups=(conv, activations_to_fuse, add)),
                       schema.Fusing(operator_groups=(fc, activations_to_fuse))]

    generated_tp = generate_base_tpc(operator_set, fusing_patterns)

    return generated_tp


def get_type(fusion):
    fusion_types = [x.type for x in fusion]
    return fusion_types


class TestLayerFusing(unittest.TestCase):
    def _compare(self, fused_nodes, expected_fusions):
        self.assertTrue(len(fused_nodes) == len(expected_fusions),
                        msg=f'Number of fusions is not as expected!')
        type_names = lambda types_list: [t.__name__ for t in types_list]
        for i, fusion in enumerate(fused_nodes):
            self.assertTrue(get_type(fusion) == expected_fusions[i] or
                            type_names(get_type(fusion)) == type_names(expected_fusions[i]),
                            msg=f'Miss-match fusion compared to expected!')

    def test_layer_fusing_1(self):
        expected_fusions = [[Conv2D, Activation]]
        model = create_network_1(INPUT_SHAPE)

        qc = QuantizationConfig(custom_tpc_opset_to_layer={"Conv": CustomOpsetLayers([Conv2D]),
                                                           "AnyReLU": CustomOpsetLayers([tf.nn.relu,
                                                                        tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                                        tp.LayerFilterParams(Activation, activation="relu")])})

        fusion_graph = prepare_graph_with_configs(model, KerasImplementation(), DEFAULT_KERAS_INFO,
                                                  representative_dataset, lambda name, _tp: get_tpc_1(),
                                                  attach2fw=AttachTpcToKeras(), qc=qc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)

    def test_layer_fusing_2(self):
        expected_fusions = [[Conv2D, Activation], [Conv2D, ReLU], [Conv2D, tf.nn.sigmoid], [Conv2D, Activation]]
        model = create_network_2(INPUT_SHAPE)

        qc = QuantizationConfig(custom_tpc_opset_to_layer={"Conv": CustomOpsetLayers([Conv2D]),
                                                           "AnyReLU": CustomOpsetLayers([tf.nn.relu,
                                                                        tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                                        tp.LayerFilterParams(Activation,
                                                                                             activation="relu")]),
                                                           "Swish": CustomOpsetLayers([tf.nn.swish, tp.LayerFilterParams(Activation,
                                                                                                        activation="swish")]),
                                                           "Sigmoid": CustomOpsetLayers([tf.nn.sigmoid, tp.LayerFilterParams(Activation,
                                                                                                            activation="sigmoid")]),
                                                           "Tanh": CustomOpsetLayers([tf.nn.tanh, tp.LayerFilterParams(Activation,
                                                                                                      activation="tanh")])})

        fusion_graph = prepare_graph_with_configs(model, KerasImplementation(), DEFAULT_KERAS_INFO,
                                                  representative_dataset, lambda name, _tp: get_tpc_2(),
                                                  attach2fw=AttachTpcToKeras(), qc=qc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)

    def test_layer_fusing_3(self):
        expected_fusions = [[Conv2D, Activation]]
        model = create_network_3(INPUT_SHAPE)

        qc = QuantizationConfig(custom_tpc_opset_to_layer={"Conv": CustomOpsetLayers([Conv2D]),
                                                           "AnyReLU": CustomOpsetLayers([tf.nn.relu,
                                                                        tp.LayerFilterParams(ReLU, negative_slope=0.0),
                                                                        tp.LayerFilterParams(Activation,
                                                                                             activation="relu")])})

        fusion_graph = prepare_graph_with_configs(model, KerasImplementation(), DEFAULT_KERAS_INFO,
                                                  representative_dataset, lambda name, _tp: get_tpc_3(),
                                                  attach2fw=AttachTpcToKeras(), qc=qc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)

    def test_layer_fusing_4(self):
        expected_fusions = [[Conv2D, Activation, Add], [Conv2D, Activation, Add], [Conv2D, Activation],
                            [Conv2D, ReLU, Add], [Dense, tf.nn.silu], [Dense, Activation]]
        model = create_network_4(INPUT_SHAPE)

        qc = QuantizationConfig(custom_tpc_opset_to_layer={
            "Conv": CustomOpsetLayers([Conv2D]),
            "FullyConnected": CustomOpsetLayers([Dense]),
            "AnyReLU": CustomOpsetLayers([tf.nn.relu,
                         tp.LayerFilterParams(ReLU, negative_slope=0.0),
                         tp.LayerFilterParams(Activation,
                                              activation="relu")]),
            "Add": CustomOpsetLayers([tf.add, Add]),
            "Swish": CustomOpsetLayers([tf.nn.swish, tp.LayerFilterParams(Activation, activation="swish")]),
        })

        fusion_graph = prepare_graph_with_configs(model, KerasImplementation(), DEFAULT_KERAS_INFO,
                                                  representative_dataset, lambda name, _tp: get_tpc_4(),
                                                  attach2fw=AttachTpcToKeras(), qc=qc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)
