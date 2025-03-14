import pytest

from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.quantization.bit_width_config import ManualBitWidthSelection, ManualWeightsBitWidthSelection
from model_compression_toolkit.core import BitWidthConfig

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from tests_pytest.test_util.graph_builder_utils import build_node

from model_compression_toolkit.target_platform_capabilities.constants import KERNEL_ATTR, BIAS_ATTR

### dummy layer classes
class Conv2D:
    pass
class InputLayer:
    pass
class Add:
    pass
class BatchNormalization:
    pass
class ReLU:
    pass
class Flatten:
    pass
class Dense:
    pass

### test model
def get_test_graph():
    n1 = build_node('input', layer_class=InputLayer)
    conv1 = build_node('conv1', layer_class=Conv2D)
    add1 = build_node('add1', layer_class=Add)
    conv2 = build_node('conv2', layer_class=Conv2D)
    bn1 = build_node('bn1', layer_class=BatchNormalization)
    relu = build_node('relu1', layer_class=ReLU)
    add2 = build_node('add2', layer_class=Add)
    flatten = build_node('flatten', layer_class=Flatten)
    fc = build_node('fc', layer_class=Dense)

    graph = Graph('g', input_nodes=[n1],
                  nodes=[conv1,add1, conv2, bn1, relu, add2, flatten],
                  output_nodes=[fc],
                  edge_list=[Edge(n1, conv1, 0, 0),
                             Edge(conv1, add1, 0, 0),
                             Edge(add1, conv2, 0, 0),
                             Edge(conv2, bn1, 0, 0),
                             Edge(bn1, relu, 0, 0),
                             Edge(relu, add2, 0, 0),
                             Edge(add1, add2, 0, 0),
                             Edge(add2, flatten, 0, 0),
                             Edge(flatten, fc, 0, 0),
                             ]
                  )
    return graph

class TestBitWidthConfig:
    # test case
    setter_test_input_0 = {"activation": (None, None),
                           "weights":    (None, None, None)}
    setter_test_input_1 = {"activation": (NodeTypeFilter(ReLU), [16]),
                           "weights":    (None, None, None)}
    setter_test_input_2 = {"activation": (None, None),
                           "weights":    (NodeNameFilter("conv2"), [8], KERNEL_ATTR)}
    setter_test_input_3 = {"activation": (NodeTypeFilter(ReLU), [16]),
                           "weights":    (NodeNameFilter("conv2"), [8], KERNEL_ATTR)}
    setter_test_input_4 = {"activation": ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16, 8]),
                           "weights":    ([NodeTypeFilter(Conv2D), NodeNameFilter("fc")], [16, 2], [KERNEL_ATTR, BIAS_ATTR])}

    setter_test_expected_0 = {"activation": (None, None),
                              "weights":    (None, None, None)}
    setter_test_expected_1 = {"activation": ([NodeTypeFilter, ReLU, 16]),
                              "weights":    (None, None, None)}
    setter_test_expected_2 = {"activation": (None, None),
                              "weights":    ([NodeNameFilter, "conv2", 8, KERNEL_ATTR]) }
    setter_test_expected_3 = {"activation": ([NodeTypeFilter, ReLU, 16]),
                              "weights":    ([NodeNameFilter, "conv2", 8, KERNEL_ATTR])}
    setter_test_expected_4 = {"activation": ([NodeTypeFilter, ReLU, 16], [NodeNameFilter, "conv1", 8]),
                              "weights":    ([NodeTypeFilter, Conv2D, 16, KERNEL_ATTR], [NodeNameFilter, "fc", 2, BIAS_ATTR])}


    # test : BitWidthConfig set_manual_activation_bit_width, set_manual_weights_bit_width
    @pytest.mark.parametrize(("inputs", "expected"), [
        (setter_test_input_0, setter_test_expected_0),
        (setter_test_input_1, setter_test_expected_1),
        (setter_test_input_2, setter_test_expected_2),
        (setter_test_input_3, setter_test_expected_3),
        (setter_test_input_4, setter_test_expected_4),
    ])
    def test_bit_width_config_setter(self, inputs, expected):

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

        def check_param_for_weights(mb_cfg, exp):
            ### check setting config class (expected ManualBitWidthSelection)
            assert type(mb_cfg) == ManualWeightsBitWidthSelection

            ### check setting filter for NodeFilter and NodeInfo
            if mb_cfg.filter is not None:
                assert isinstance(mb_cfg.filter, exp[0])
                if isinstance(mb_cfg.filter, NodeTypeFilter):
                    assert mb_cfg.filter.node_type == exp[1]
                elif isinstance(mb_cfg.filter, NodeNameFilter):
                    assert mb_cfg.filter.node_name == exp[1]

                ### check setting bit_width
                assert mb_cfg.bit_width == exp[2]
                assert mb_cfg.attr == exp[3]
            else:
                assert mb_cfg.filter is None

        activation = inputs["activation"]
        weights = inputs["weights"]

        activation_expected = expected["activation"]
        weights_expected = expected["weights"]

        manual_bit_cfg = BitWidthConfig()

        manual_bit_cfg.set_manual_activation_bit_width(activation[0], activation[1])
        manual_bit_cfg.set_manual_weights_bit_width(weights[0], weights[1], weights[2])

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
                check_param_for_weights(w_mb_cfg, weights_expected)
        else:
            for idx, w_mb_cfg in enumerate(manual_bit_cfg.manual_weights_bit_width_selection_list):
                check_param_for_weights(w_mb_cfg, weights_expected[idx])


    ### test case
    ### Note: setter inputs reuse getters test inputs
    getter_test_expected_0 = {"activation":{},
                              "weights": {}}
    getter_test_expected_1 = {"activation":{"ReLU:relu1": 16},
                              "weights": {}}
    getter_test_expected_2 = {"activation":{},
                              "weights": {"Conv2D:conv2": [8, KERNEL_ATTR]}}
    getter_test_expected_3 = {"activation": {"ReLU:relu1": 16},
                              "weights": {"Conv2D:conv2": [8, KERNEL_ATTR]}}
    getter_test_expected_4 = {"activation": {"ReLU:relu1": 16, "Conv2D:conv1": 8},
                              "weights": {"Conv2D:conv1": [16, KERNEL_ATTR], "Conv2D:conv2": [16, KERNEL_ATTR], "Dense:fc": [2, BIAS_ATTR]}}

    # test : BitWidthConfig get_nodes_to_manipulate_bit_widths
    @pytest.mark.parametrize(("inputs", "expected"), [
        (setter_test_input_0, getter_test_expected_0),
        (setter_test_input_1, getter_test_expected_1),
        (setter_test_input_2, getter_test_expected_2),
        (setter_test_input_3, getter_test_expected_3),
        (setter_test_input_4, getter_test_expected_4),
    ])
    def test_bit_width_config_getter(self, inputs, expected):

        graph = get_test_graph()

        activation = inputs["activation"]
        weights = inputs["weights"]

        activation_expected = expected["activation"]
        weights_expected = expected["weights"]

        manual_bit_cfg = BitWidthConfig()
        if activation[0] is not None:
            manual_bit_cfg.set_manual_activation_bit_width(activation[0], activation[1])
        if weights[0] is not None:
            manual_bit_cfg.set_manual_weights_bit_width(weights[0], weights[1], weights[2])

        get_manual_bit_dict_activation = manual_bit_cfg.get_nodes_to_manipulate_activation_bit_widths(graph)
        get_manual_bit_dict_weights = manual_bit_cfg.get_nodes_to_manipulate_weights_bit_widths(graph)

        if activation[0] is not None:
            for idx, (key, val) in enumerate(get_manual_bit_dict_activation.items()):
                assert str(key) == list(activation_expected.keys())[idx]
                assert val == list(activation_expected.values())[idx]
        else:
            assert get_manual_bit_dict_activation == activation_expected

        if weights[0] is not None:
            for idx, (key, val) in enumerate(get_manual_bit_dict_weights.items()):
                assert str(key) == list(weights_expected.keys())[idx]
                assert val[0] == list(weights_expected.values())[idx][0]
                assert val[1] == list(weights_expected.values())[idx][1]
        else:
            assert get_manual_bit_dict_weights == weights_expected
