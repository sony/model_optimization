import pytest

from model_compression_toolkit.core.common.network_editors import NodeTypeFilter, NodeNameFilter
from model_compression_toolkit.core.common.quantization.bit_width_config import ManualBitWidthSelection, ManualWeightsBitWidthSelection
from model_compression_toolkit.core import BitWidthConfig

from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.graph.edge import Edge
from tests_pytest._test_util.graph_builder_utils import build_node


TEST_KERNEL = 'kernel'
TEST_BIAS = 'bias'

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
    conv1 = build_node('conv1', canonical_weights={TEST_KERNEL: [1, 2], TEST_BIAS: [3, 4]}, layer_class=Conv2D)
    add1 = build_node('add1', layer_class=Add)
    conv2 = build_node('conv2', layer_class=Conv2D)
    bn1 = build_node('bn1', layer_class=BatchNormalization)
    relu = build_node('relu1', canonical_weights={TEST_KERNEL: [1, 2], TEST_BIAS: [3, 4]}, layer_class=ReLU)
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


@pytest.skip("TODO manual bitwidth unittest", allow_module_level=True)
class TestBitWidthConfig:
    # test case for set_manual_activation_bit_width
    test_input_0 = (None, None)
    test_input_1 = (NodeTypeFilter(ReLU), 16)
    test_input_2 = ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16])
    test_input_3 = ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16, 8])

    test_expected_0 = ("The filters cannot be None.", None)
    test_expected_1 = (NodeTypeFilter, ReLU, 16)
    test_expected_2 = ([NodeTypeFilter, ReLU, 16], [NodeNameFilter, "conv1", 16])
    test_expected_3 = ([NodeTypeFilter, ReLU, 16], [NodeNameFilter, "conv1", 8])

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_0, test_expected_0),
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_set_manual_activation_bit_width(self, inputs, expected):
        def check_param_for_activation(mb_cfg, exp):
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

        manual_bit_cfg = BitWidthConfig()
        try:
            manual_bit_cfg.set_manual_activation_bit_width(inputs[0], inputs[1])
            ### check Activation
            if len(manual_bit_cfg.manual_activation_bit_width_selection_list) == 1:
                for a_mb_cfg in manual_bit_cfg.manual_activation_bit_width_selection_list:
                    print(a_mb_cfg, expected)
                    check_param_for_activation(a_mb_cfg, expected)
            else:
                for idx, a_mb_cfg in enumerate(manual_bit_cfg.manual_activation_bit_width_selection_list):
                    check_param_for_activation(a_mb_cfg, expected[idx])
        except Exception as e:
            assert str(e) == expected[0]


    # test case for set_manual_weights_bit_width
    test_input_0 = (None, None, None)
    test_input_1 = (NodeTypeFilter(ReLU), 16, TEST_KERNEL)
    test_input_2 = ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16], [TEST_KERNEL])
    test_input_3 = ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16, 8], [TEST_KERNEL, TEST_BIAS])

    test_expected_0 = ("The filters cannot be None.", None, None)
    test_expected_1 = (NodeTypeFilter, ReLU, 16, TEST_KERNEL)
    test_expected_2 = ([NodeTypeFilter, ReLU, 16, TEST_KERNEL], [NodeNameFilter, "conv1", 16, TEST_KERNEL])
    test_expected_3 = ([NodeTypeFilter, ReLU, 16, TEST_KERNEL], [NodeNameFilter, "conv1", 8, TEST_BIAS])

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_0, test_expected_0),
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_set_manual_weights_bit_width(self, inputs, expected):
        def check_param_weights(mb_cfg, exp):
            ### check setting config class (expected ManualWeightsBitWidthSelection)
            assert type(mb_cfg) == ManualWeightsBitWidthSelection

            ### check setting filter for NodeFilter and NodeInfo
            if mb_cfg.filter is not None:
                assert isinstance(mb_cfg.filter, exp[0])
                if isinstance(mb_cfg.filter, NodeTypeFilter):
                    assert mb_cfg.filter.node_type == exp[1]
                elif isinstance(mb_cfg.filter, NodeNameFilter):
                    assert mb_cfg.filter.node_name == exp[1]

                ### check setting bit_width and attr
                assert mb_cfg.bit_width == exp[2]
                assert mb_cfg.attr == exp[3]
            else:
                assert mb_cfg.filter is None

        manual_bit_cfg = BitWidthConfig()
        try:
            manual_bit_cfg.set_manual_weights_bit_width(inputs[0], inputs[1], inputs[2])
            ### check weights
            if len(manual_bit_cfg.manual_weights_bit_width_selection_list) == 1:
                for a_mb_cfg in manual_bit_cfg.manual_weights_bit_width_selection_list:
                    print(a_mb_cfg, expected)
                    check_param_weights(a_mb_cfg, expected)
            else:
                for idx, a_mb_cfg in enumerate(manual_bit_cfg.manual_weights_bit_width_selection_list):
                    check_param_weights(a_mb_cfg, expected[idx])
        except Exception as e:
            assert str(e) == expected[0]


    # test case for get_nodes_to_manipulate_activation_bit_widths
    test_input_0 = (NodeTypeFilter(ReLU), 16)
    test_input_1 = (NodeNameFilter('relu1'), 16)
    test_input_2 = ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16, 8])

    test_expected_0 = ({"ReLU:relu1": 16})
    test_expected_1 = ({"ReLU:relu1": 16})
    test_expected_2 = ({"ReLU:relu1": 16, "Conv2D:conv1": 8})

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_0, test_expected_0),
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
    ])
    def test_get_nodes_to_manipulate_activation_bit_widths(self, inputs, expected):
        fl_list = inputs[0] if isinstance(inputs[0], list) else [inputs[0]]
        bw_list = inputs[1] if isinstance(inputs[1], list) else [inputs[1]]

        mbws_config = []
        for fl, bw in zip(fl_list, bw_list):
            mbws_config.append(ManualBitWidthSelection(fl, bw))
        manual_bit_cfg = BitWidthConfig(manual_activation_bit_width_selection_list=mbws_config)

        graph = get_test_graph()
        get_manual_bit_dict_activation = manual_bit_cfg.get_nodes_activation_bit_widths(graph)
        for idx, (key, val) in enumerate(get_manual_bit_dict_activation.items()):
            assert str(key) == list(expected.keys())[idx]
            assert val == list(expected.values())[idx]


    # test case for get_nodes_to_manipulate_weights_bit_widths
    test_input_0 = (NodeTypeFilter(ReLU), 16, TEST_KERNEL)
    test_input_1 = (NodeNameFilter('relu1'), 16, TEST_BIAS)
    test_input_2 = ([NodeTypeFilter(ReLU), NodeNameFilter("conv1")], [16, 8], [TEST_KERNEL, TEST_BIAS])
    test_input_3 = ([NodeNameFilter("conv1"), NodeNameFilter("conv1")], [4, 8], [TEST_KERNEL, TEST_BIAS])

    test_expected_0 = ({"ReLU:relu1": [[16, TEST_KERNEL]]})
    test_expected_1 = ({"ReLU:relu1": [[16, TEST_BIAS]]})
    test_expected_2 = ({"ReLU:relu1": [[16, TEST_KERNEL]], "Conv2D:conv1": [[8, TEST_BIAS]]})
    test_expected_3 = ({"Conv2D:conv1": [[4, TEST_KERNEL], [8, TEST_BIAS]]})

    @pytest.mark.parametrize(("inputs", "expected"), [
        (test_input_0, test_expected_0),
        (test_input_1, test_expected_1),
        (test_input_2, test_expected_2),
        (test_input_3, test_expected_3),
    ])
    def test_get_nodes_to_manipulate_weights_bit_widths(self, inputs, expected):
        fl_list = inputs[0] if isinstance(inputs[0], list) else [inputs[0]]
        bw_list = inputs[1] if isinstance(inputs[1], list) else [inputs[1]]
        at_list = inputs[2] if isinstance(inputs[2], list) else [inputs[2]]

        manual_weights_bit_width_config = []
        for fl, bw, at in zip(fl_list, bw_list, at_list):
            manual_weights_bit_width_config.append(ManualWeightsBitWidthSelection(fl, bw, at))
        manual_bit_cfg = BitWidthConfig(manual_weights_bit_width_selection_list=manual_weights_bit_width_config)

        graph = get_test_graph()
        get_manual_bit_dict_weights = manual_bit_cfg.get_nodes_weights_bit_widths(graph)
        for idx, (key, val) in enumerate(get_manual_bit_dict_weights.items()):
            assert str(key) == list(expected.keys())[idx]
            assert val == list(expected.values())[idx]
