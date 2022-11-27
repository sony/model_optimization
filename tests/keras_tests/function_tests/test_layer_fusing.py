import unittest
import numpy as np
import tensorflow as tf

from model_compression_toolkit import DEFAULTCONFIG
from model_compression_toolkit.core.common.fusion.layer_fusing import fusion
from model_compression_toolkit.core.common.quantization.set_node_quantization_config import \
    set_quantization_configuration_to_graph
from model_compression_toolkit.core.common.substitutions.apply_substitutions import substitute
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.core.tpc_models.default_tpc.latest import get_op_quantization_configs
import model_compression_toolkit as mct
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


def generate_base_tpc():
    default_config, mixed_precision_cfg_list = get_op_quantization_configs()
    default_configuration_options = tp.QuantizationConfigOptions([default_config])
    generated_tp = tp.TargetPlatformModel(default_configuration_options, name='layer_fusing_test')
    mixed_precision_configuration_options = tp.QuantizationConfigOptions(mixed_precision_cfg_list,
                                                                         base_config=default_config)

    return generated_tp, mixed_precision_configuration_options


def get_tpc_1():
    generated_tp, mixed_precision_configuration_options = generate_base_tpc()
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


def get_tpc_2():
    generated_tp, mixed_precision_configuration_options = generate_base_tpc()
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


def get_tpc_3():
    generated_tp, mixed_precision_configuration_options = generate_base_tpc()
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


def get_tpc_4():
    generated_tp, mixed_precision_configuration_options = generate_base_tpc()
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


def get_type(fusion):
    fusion_types = [x.type for x in fusion]
    return fusion_types


def prepare_graph(in_model, tpc):
    fw_info = DEFAULT_KERAS_INFO
    qc = DEFAULTCONFIG
    keras_impl = KerasImplementation()
    graph = keras_impl.model_reader(in_model, representative_dataset)

    graph.set_fw_info(fw_info)
    graph.set_tpc(tpc)

    # Standard graph substitutions
    graph = substitute(graph, keras_impl.get_substitutions_prepare_graph())
    for node in graph.nodes:
        node.prior_info = keras_impl.get_node_prior_info(node=node,
                                                         fw_info=fw_info, graph=graph)
    graph = substitute(graph, keras_impl.get_substitutions_pre_statistics_collection(qc))

    graph = set_quantization_configuration_to_graph(graph=graph,
                                                    quant_config=qc,
                                                    mixed_precision_enable=True)
    fusion_graph = fusion(graph, tpc)

    return fusion_graph


class TestLayerFusing(unittest.TestCase):
    def _compare(self, fused_nodes, expected_fusions):
        self.assertTrue(len(fused_nodes) == len(expected_fusions),
                        msg=f'Number of fusions is not as expected!')
        for i, fusion in enumerate(fused_nodes):
            self.assertTrue(get_type(fusion) == expected_fusions[i],
                            msg=f'Miss-match fusion compared to expected!')

    def test_layer_fusing_1(self):
        expected_fusions = [[Conv2D, Activation]]
        tpc = get_tpc_1()
        model = create_network_1(INPUT_SHAPE)

        fusion_graph = prepare_graph(model, tpc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)

    def test_layer_fusing_2(self):
        expected_fusions = [[Conv2D, Activation], [Conv2D, ReLU], [Conv2D, tf.nn.sigmoid], [Conv2D, Activation]]
        tpc = get_tpc_2()
        model = create_network_2(INPUT_SHAPE)

        fusion_graph = prepare_graph(model, tpc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)

    def test_layer_fusing_3(self):
        expected_fusions = [[Conv2D, Activation]]
        tpc = get_tpc_3()
        model = create_network_3(INPUT_SHAPE)

        fusion_graph = prepare_graph(model, tpc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)

    def test_layer_fusing_4(self):
        expected_fusions = [[Conv2D, Activation, Add], [Conv2D, Activation, Add], [Conv2D, Activation], [Conv2D, ReLU, Add], [Dense, tf.nn.silu], [Dense, Activation]]
        tpc = get_tpc_4()
        model = create_network_4(INPUT_SHAPE)

        fusion_graph = prepare_graph(model, tpc)

        self._compare(fusion_graph.fused_nodes, expected_fusions)
