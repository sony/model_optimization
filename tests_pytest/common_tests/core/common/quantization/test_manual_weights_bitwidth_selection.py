import pytest
import numpy as np
import tensorflow as tf

import model_compression_toolkit as mct

from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.quantization.bit_width_config import ManualBitWidthSelection
from model_compression_toolkit.core import BitWidthConfig


tf.config.set_visible_devices([], 'GPU')

### test model
def get_test_model():
    ### Note: This test model is ref for mct
    ### path: model_optimization/tests/keras_tests/feature_networks_tests/feature_networks/manual_bit_selection.py

    input_tensor = tf.keras.layers.Input(shape=(224,224,3), name='input')
    x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='conv1')(input_tensor)
    x1 = tf.keras.layers.Add(name='add1')([x1, np.ones((3,), dtype=np.float32)])

    # Second convolutional block
    x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', name='conv2')(x1)
    x2 = tf.keras.layers.BatchNormalization(name='bn1')(x2)
    x2 = tf.keras.layers.ReLU(name='relu1')(x2)

    # Addition
    x = tf.keras.layers.Add(name='add2')([x1, x2])

    # Flatten and fully connected layer
    x = tf.keras.layers.Flatten()(x)
    output_tensor = tf.keras.layers.Dense(units=10, activation='softmax', name='fc')(x)

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

def get_test_graph(model):

    from model_compression_toolkit.core.graph_prep_runner import read_model_to_graph
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation

    from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import AttachTpcToKeras
    from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities


    fw_info = DEFAULT_KERAS_INFO
    fw_impl = KerasImplementation()

    target_platform_capabilities = mct.get_target_platform_capabilities(fw_name='tensorflow',
                                                                        target_platform_name='imx500',
                                                                        target_platform_version='v1')

    attach2keras = AttachTpcToKeras()
    target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
    framework_platform_capabilities = attach2keras.attach(target_platform_capabilities)

    graph = read_model_to_graph(model,
                                None,
                                framework_platform_capabilities,
                                fw_info, fw_impl)

    return graph

class TestBitWidthConfig:

    #######################################################################################################
    ### test case
    setter_test_input_0 = {"activation": (None, None),
                           "weights":    (None, None)}
    setter_test_input_1 = {"activation": (NodeTypeFilter(tf.keras.layers.ReLU), [16]),
                           "weights":    (None, None)}
    setter_test_input_2 = {"activation": (None, None),
                           "weights":    (NodeNameFilter("conv2"), [8])}
    setter_test_input_3 = {"activation": (NodeTypeFilter(tf.keras.layers.ReLU), [16]),
                           "weights":    (NodeNameFilter("conv2"), [8])}
    setter_test_input_4 = {"activation": ([NodeTypeFilter(tf.keras.layers.ReLU), NodeNameFilter("conv1")], [16, 8]),
                           "weights":    ([NodeTypeFilter(tf.keras.layers.Conv2D), NodeNameFilter("fc")], [16, 2])}

    setter_test_expected_0 = {"activation": (None, None),
                              "weights":    (None, None)}
    setter_test_expected_1 = {"activation": ([NodeTypeFilter, tf.keras.layers.ReLU, 16]),
                              "weights":    (None, None)}
    setter_test_expected_2 = {"activation": (None, None),
                              "weights":    ([NodeNameFilter, "conv2", 8]) }
    setter_test_expected_3 = {"activation": ([NodeTypeFilter, tf.keras.layers.ReLU, 16]),
                              "weights":    ([NodeNameFilter, "conv2", 8])}
    setter_test_expected_4 = {"activation": ([NodeTypeFilter, tf.keras.layers.ReLU, 16], [NodeNameFilter, "conv1", 8]),
                              "weights":    ([NodeTypeFilter, tf.keras.layers.Conv2D, 16], [NodeNameFilter, "fc", 2])}


    # test : BitWidthConfig set_manual_activation_bit_width, set_manual_weights_bit_width
    @pytest.mark.parametrize(("inputs", "expected"), [
        (setter_test_input_0, setter_test_expected_0),
        (setter_test_input_1, setter_test_expected_1),
        (setter_test_input_2, setter_test_expected_2),
        (setter_test_input_3, setter_test_expected_3),
        (setter_test_input_4, setter_test_expected_4),
    ])
    def test_BitWidthConfig_setter(self, inputs, expected):

        def check_param(mb_cfg, exp):
            ### check setting config class (expected ManualBitWidthSelection)
            assert type(mb_cfg) == ManualBitWidthSelection

            ### check setting filter for NodeFilter and NodeInfo
            if mb_cfg.filter is not None:
                assert isinstance(mb_cfg.filter, exp[0])
                if isinstance(mb_cfg.filter, NodeTypeFilter):
                    assert mb_cfg.filter.node_type == exp[1]
                elif isinstance(mb_cfg.filter, NodeNameFilter):
                    assert mb_cfg.filter.node_name == exp[1]

                ### check setting bit_width
                assert mb_cfg.bit_width == exp[2]
            else:
                assert mb_cfg.filter is None


        activation = inputs["activation"]
        weights = inputs["weights"]

        activation_expected = expected["activation"]
        weights_expected = expected["weights"]

        manual_bit_cfg = BitWidthConfig()

        manual_bit_cfg.set_manual_activation_bit_width(activation[0], activation[1])
        manual_bit_cfg.set_manual_weights_bit_width(weights[0], weights[1])

        ### check got object instance
        assert isinstance(manual_bit_cfg, BitWidthConfig)

        ### check Activation
        if len(manual_bit_cfg.manual_activation_bit_width_selection_list) == 1:
            for a_mb_cfg in manual_bit_cfg.manual_activation_bit_width_selection_list:
                check_param(a_mb_cfg, activation_expected)
        else:
            for idx, a_mb_cfg in enumerate(manual_bit_cfg.manual_activation_bit_width_selection_list):
                check_param(a_mb_cfg, activation_expected[idx])

        ### check Weights
        if len(manual_bit_cfg.manual_weights_bit_width_selection_list) == 1:
            for w_mb_cfg in manual_bit_cfg.manual_weights_bit_width_selection_list:
                check_param(w_mb_cfg, weights_expected)
        else:
            for idx, w_mb_cfg in enumerate(manual_bit_cfg.manual_weights_bit_width_selection_list):
                check_param(w_mb_cfg, weights_expected[idx])


    #######################################################################################################
    ### test case
    ### Note: setter inputs reuse getters test inputs
    getter_test_expected_0 = {"activation":{},
                              "weights": {}}
    getter_test_expected_1 = {"activation":{"ReLU:relu1": 16},
                              "weights": {}}
    getter_test_expected_2 = {"activation":{},
                              "weights": {"Conv2D:conv2": 8}}
    getter_test_expected_3 = {"activation": {"ReLU:relu1": 16},
                              "weights": {"Conv2D:conv2": 8}}
    getter_test_expected_4 = {"activation": {"ReLU:relu1": 16, "Conv2D:conv1": 8},
                              "weights": {"Conv2D:conv2": 16, "Conv2D:conv1": 16, "Dense:fc": 2}}

    # test : BitWidthConfig get_nodes_to_manipulate_bit_widths
    @pytest.mark.parametrize(("model", "inputs", "expected"), [
        (get_test_model(), setter_test_input_0, getter_test_expected_0),
        (get_test_model(), setter_test_input_1, getter_test_expected_1),
        (get_test_model(), setter_test_input_2, getter_test_expected_2),
        (get_test_model(), setter_test_input_3, getter_test_expected_3),
        (get_test_model(), setter_test_input_4, getter_test_expected_4),
    ])
    def test_BitWidthConfig_getter(self, model, inputs, expected):

        graph = get_test_graph(model)

        activation = inputs["activation"]
        weights = inputs["weights"]

        activation_expected = expected["activation"]
        weights_expected = expected["weights"]

        manual_bit_cfg = BitWidthConfig()
        if activation[0] is not None:
            manual_bit_cfg.set_manual_activation_bit_width(activation[0], activation[1])
        if weights[0] is not None:
            manual_bit_cfg.set_manual_weights_bit_width(weights[0], weights[1])

        get_manual_bit_dict = manual_bit_cfg.get_nodes_to_manipulate_bit_widths(graph)

        if activation[0] is not None:
            for idx, (key, val) in enumerate(get_manual_bit_dict["activation"].items()):
                assert str(key) == list(activation_expected.keys())[idx]
                assert val == list(activation_expected.values())[idx]
        else:
            assert get_manual_bit_dict["activation"] == activation_expected

        if weights[0] is not None:
            for idx, (key, val) in enumerate(get_manual_bit_dict["weights"].items()):
                assert str(key) == list(weights_expected.keys())[idx]
                assert val == list(weights_expected.values())[idx]
        else:
            assert get_manual_bit_dict["weights"] == weights_expected
