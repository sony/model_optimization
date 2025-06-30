# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy

import pytest
from unittest.mock import Mock, call

import numpy as np

from model_compression_toolkit.core import ResourceUtilization, MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_info import set_fw_info
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.graph.virtual_activation_weights_node import VirtualActivationWeightsNode, \
    VirtualSplitActivationNode, VirtualSplitWeightsNode
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MpMetricNormalization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_ru_helper import MixedPrecisionRUHelper
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager, ConfigReconstructionHelper
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.sensitivity_eval.sensitivity_evaluation import SensitivityEvaluation
from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc, DummyLayer


class DummyLayer1:
    pass


class DummyLayer2:
    pass


def build_graph(fw_info_mock, w_mp: bool, a_mp: bool):
    # activations sizeX[nbits]:  6 x [8],  120 x [2, 8, 4],  40 x [4, 8],    273 x [2],       34 x [8]
    # weights:                                               42 x [4, 8, 2], 142 x [12, 4, 8]
    # cuts max: [48, 1008, 1280, 866, 818, 272] ([6*8, 6*8+120*8, 120*8+40*8, 40*8+273*2, 273*2+34*8, 34*8])
    # cuts min: [48, 288, 400, 706, 818, 272] ([6*8, 6*8+120*2, 120*2+40*4, 40*4+273*2, 273*2+34*8, 34*8])
    # n2 qcs: [a8, a2, a4]
    # n3 qcs: [a4w4, a4w8, a4w2, a8w4, a8w8, a8w2]
    # n4 qcs: [w12, w4, w8]
    op_kernels = {DummyLayer1: 'foo', DummyLayer2: 'bar'}
    fw_info_mock.get_kernel_op_attribute = lambda nt: op_kernels.get(nt)

    n1 = build_node('n1', qcs=[build_nbits_qc(a_nbits=8)], output_shape=(None, 1, 2, 3))

    abits2 = [8, 2, 4] if a_mp else [4]
    n2 = build_node('n2', qcs=[build_nbits_qc(a_nbits=ab) for ab in abits2], output_shape=(None, 4, 5, 6))

    abit3 = [4, 8] if a_mp else [8]
    wbit3 = [4, 8, 2] if w_mp else [4]
    n3 = build_node('n3', canonical_weights={'foo': np.ones((3, 14))},
                    qcs=[build_nbits_qc(a_nbits=ab, w_attr={'foo': (wb, True)}) for ab in abit3 for wb in wbit3],
                    output_shape=(None, 5, 8), layer_class=DummyLayer1)

    wbit4 = [12, 4, 8] if w_mp else [8]
    n4 = build_node('n4', canonical_weights={'bar': np.ones((2, 71))},
                    qcs=[build_nbits_qc(a_nbits=2, w_attr={'bar': (wb, True)}) for wb in wbit4],
                    output_shape=(None, 13, 21), layer_class=DummyLayer2)
    n5 = build_node('n5', qcs=[build_nbits_qc(a_nbits=8)], output_shape=(None, 34, 1))

    g = Graph('g', input_nodes=[n1], nodes=[n2, n3, n4], output_nodes=[n5],
              edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0), Edge(n3, n4, 0, 0), Edge(n4, n5, 0, 0)])
    return g, [n1, n2, n3, n4, n5]


class TestMixedPrecisionSearchManager:
    """ MP search manager tests.
        TODO: Sensitivity computation is not tested.
              BOPS: only logical flow is tested.
    """
    @pytest.fixture(autouse=True)
    def setup(self, fw_info_mock):
        set_fw_info(fw_info_mock)

    def test_prepare_weights_ru_for_lp(self, fw_info_mock, fw_impl_mock):
        """ Tests ru related setup and methods for weights target. """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=True, a_mp=False)
        ru = ResourceUtilization(weights_memory=100)
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(), target_resource_utilization=ru,
                                          mp_config=Mock())
        assert mgr.min_ru_config == {n3: 2, n4: 1}
        assert mgr.min_ru == {RUTarget.WEIGHTS: 3 * 14 * 2 / 8 + 2 * 71 * 4 / 8}

        rel_ru = mgr._compute_relative_ru_matrices()
        self._assert_dict_allclose(rel_ru, {RUTarget.WEIGHTS: np.array([2*42, 6*42, 0, 8*142, 0, 4*142])[:, None]/8})
        rel_constraint = mgr._get_relative_ru_constraint_per_mem_element()
        self._assert_dict_allclose(rel_constraint, {RUTarget.WEIGHTS: np.array([[100 - 81.5]])})

    def test_prepare_activation_ru_for_lp(self, fw_info_mock, fw_impl_mock):
        """ Tests ru related setup and methods for activation target. """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=False, a_mp=True)
        ru = ResourceUtilization(activation_memory=150)
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(), target_resource_utilization=ru,
                                          mp_config=Mock())
        # 6 x [8], 120 x [8, 2, 4], 40 x [4, 8], 273 x [2], 34 x [8]
        assert mgr.min_ru_config == {n2: 1, n3: 0}
        self._assert_dict_allclose(mgr.min_ru,
                                   {RUTarget.ACTIVATION: np.array([48, 48+240, 240+160, 160+546, 546+272, 272])/8},
                                   sort_axis=0)

        rel_ru = mgr._compute_relative_ru_matrices()    # candidates X cuts
        self._assert_dict_allclose(rel_ru,
                                   {RUTarget.ACTIVATION: np.array([[0, 120*6, 120*6, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0],
                                                                   [0, 120*2, 120*2, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0],
                                                                   [0, 0, 40*4, 40*4, 0, 0]])/8},
                                   sort_axis=1)

        rel_constraint = mgr._get_relative_ru_constraint_per_mem_element()
        self._assert_dict_allclose(rel_constraint,
                                   {RUTarget.ACTIVATION: 150 - np.array([48, 288, 400, 706, 818, 272])/8},
                                   sort_axis=0)

    def test_prepare_total_ru_for_lp(self, fw_info_mock, fw_impl_mock):
        """ Tests ru related setup and methods for total target.  """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=True, a_mp=True)
        ru = ResourceUtilization(total_memory=200)
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(), target_resource_utilization=ru,
                                          mp_config=Mock())
        assert mgr.min_ru_config == {n2: 1, n3: 2, n4: 1}

        self._assert_dict_allclose(mgr.min_ru,
                                   {RUTarget.TOTAL: 81.5 + np.array([48, 288, 400, 706, 818, 272]) / 8},
                                   sort_axis=0)

        rel_ru = mgr._compute_relative_ru_matrices()
        self._assert_dict_allclose(rel_ru,
                                   {RUTarget.TOTAL: np.array([[0, 120*6, 120*6, 0, 0, 0],              # n2
                                                              [0, 0, 0, 0, 0, 0],                      # n2
                                                              [0, 120*2, 120*2, 0, 0, 0],              # n2
                                                              [42*2, 42*2, 42*2, 42*2, 42*2, 42*2],    # n3 a4w4
                                                              [42*6, 42*6, 42*6, 42*6, 42*6, 42*6],    # n3 a4w8
                                                              [0, 0, 0, 0, 0, 0],                      # n3 a4w2
                                                              [42*2, 42*2, 40*4+42*2, 40*4+42*2, 42*2, 42*2],  # n3 a8w4
                                                              [42*6, 42*6, 40*4+42*6, 40*4+42*6, 42*6, 42*6],  # n3 a8w8
                                                              [0, 0, 40*4, 40*4, 0, 0],                        # n3 a8w2
                                                              [142*8, 142*8, 142*8, 142*8, 142*8, 142*8],      # n4
                                                              [0, 0, 0, 0, 0, 0],                              # n4
                                                              [142*4, 142*4, 142*4, 142*4, 142*4, 142*4]])/8}, # n4
                                   sort_axis=1)

        rel_constraint = mgr._get_relative_ru_constraint_per_mem_element()
        self._assert_dict_allclose(rel_constraint,
                                   {RUTarget.TOTAL: 200 - 81.5 - np.array([48, 288, 400, 706, 818, 272]) / 8},
                                   sort_axis=0)

    def test_compute_relative_ru_virtual_graph(self, mocker, graph_mock, fw_impl_mock):
        """ Test compute ru mapping for virtual graph. Here we mostly test apis integration.
            - mock virtual graph, config reconstructor and ru helper compute_utilization
            - config reconstructor is called with correct args
            - compute_utilization is called on reconstructed configs
            - compute_utilization results are aggregated correctly
        """
        # mock virtual graph
        vg_mock = Mock()
        v_nodes = [Mock(candidates_quantization_cfg=[Mock(), Mock(), Mock()]),
                   Mock(candidates_quantization_cfg=[Mock(), Mock()])]
        vg_mock.get_configurable_sorted_nodes = lambda *args: v_nodes
        vg_mock.get_min_candidates_config = Mock(return_value={v_nodes[0]: 1, v_nodes[1]: 0})
        mocker.patch.object(MixedPrecisionSearchManager, '_get_mp_graph', Mock(return_value=(vg_mock, True)))

        # mock compute_utilization - first call is in ctor for minru
        compute_ru_mock = mocker.patch.object(MixedPrecisionRUHelper, 'compute_utilization',
                                              Mock(side_effect=[{RUTarget.BOPS: 5, RUTarget.TOTAL: 10},
                                                                {RUTarget.BOPS: 11, RUTarget.TOTAL: 12},
                                                                {RUTarget.BOPS: 13, RUTarget.TOTAL: 14},
                                                                {RUTarget.BOPS: 15, RUTarget.TOTAL: 16}]))
        # mock reconstructed configs
        reconstructed_cfgs = [Mock(), Mock(), Mock(), Mock()]
        recon_cfg_mock = mocker.patch.object(ConfigReconstructionHelper, 'reconstruct_full_configuration',
                                             Mock(side_effect=reconstructed_cfgs))
        mgr = MixedPrecisionSearchManager(graph_mock, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ResourceUtilization(bops=100, total_memory=100),
                                          mp_config=Mock())
        ru = mgr._compute_relative_ru_matrices()

        # first call is from c'tor for min_ru. After that we only compute non-min configurations.
        # Note: min config is copied inside, but it's a shallow copy so the node objects remain the same
        assert recon_cfg_mock.call_args_list == [
            call({v_nodes[0]: 1, v_nodes[1]: 0}),
            call({v_nodes[0]: 0, v_nodes[1]: 0}),
            call({v_nodes[0]: 2, v_nodes[1]: 0}),
            call({v_nodes[0]: 1, v_nodes[1]: 1})
        ]
        assert compute_ru_mock.call_args_list == [call({RUTarget.BOPS, RUTarget.TOTAL}, rcfg)
                                                  for rcfg in reconstructed_cfgs]
        assert len(ru) == 2
        assert np.array_equal(ru[RUTarget.BOPS], np.array([6, 0, 8, 0, 10]))
        assert np.array_equal(ru[RUTarget.TOTAL], np.array([2, 0, 4, 0, 6]))

    def test_search_weights(self, fw_info_mock, fw_impl_mock):
        """ Tests mp search with weights ru constraint.  """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=True, a_mp=False)

        def run(w_mem, sensitivity, exp_cfg):
            self._run_search_test(g, ResourceUtilization(weights_memory=w_mem), sensitivity, exp_cfg, fw_impl_mock)

        # max config solution, tight constraint
        run(w_mem=42*8/8+142*12/8, sensitivity={n3: [2, 1, 3], n4: [4, 5, 6]}, exp_cfg={n3: 1, n4: 0})

        # next best solution (that increases sensitivity the least)
        run(w_mem=42+142*12/8-1, sensitivity={n3: [2, 1, 4], n4: [1, 4, 4]}, exp_cfg={n3: 0, n4: 0})

        # min config solution, min ru constraint
        run(w_mem=42*2/8+142*4/8, sensitivity={n3: [2, 1, 100], n4: [4, 100, 6]}, exp_cfg={n3: 2, n4: 1})

        with pytest.raises(ValueError, match='model cannot be quantized to meet the specified resource utilization'):
            run(w_mem=42*2/8+142*4/8-1, sensitivity={n2: [1, 1, 1], n3: [1, 1, 1]}, exp_cfg=None)

    def test_search_activation(self, fw_info_mock, fw_impl_mock):
        """ Tests mp search with activation ru constraint.  """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=False, a_mp=True)

        def run(a_mem, sensitivity, exp_cfg):
            return self._run_search_test(g, ResourceUtilization(activation_memory=a_mem), sensitivity, exp_cfg, fw_impl_mock)

        # max config solution, tight max cut ru constraint
        res, mgr = run(a_mem=1280/8, sensitivity={n2: [1, 2, 3], n3: [5, 4]}, exp_cfg={n2: 0, n3: 1})
        assert res == {n2: 0, n3: 1}

        run(a_mem=1280/8-1, sensitivity={n2: [1, 2, 4], n3: [6, 4]}, exp_cfg={n2: 1, n3: 1})

        # min config solution, min ru constraint
        res, mgr = run(a_mem=818/8, sensitivity={n2: [1, 2, 3], n3: [100, 4]}, exp_cfg={n2: 1, n3: 0})
        assert res == mgr.min_ru_config

        with pytest.raises(ValueError, match='model cannot be quantized to meet the specified resource utilization'):
            run(a_mem=818/8-1, sensitivity={n2: [1, 1, 1], n3: [1, 1]}, exp_cfg=None)

    def test_search_all_mem(self, fw_info_mock, fw_impl_mock):
        """ Tests mp search with weights, activation and total constraints.  """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=True, a_mp=True)

        def run(ru, sensitivity, exp_cfg):
            return self._run_search_test(g, ru, sensitivity, exp_cfg, fw_impl_mock)

        # max config tight ru
        ru = ResourceUtilization(activation_memory=160, weights_memory=255, total_memory=415)
        sensitivity = {n2: [1, 2, 4], n3: [4, 4, 4, 4, 1, 4], n4: [1, 4, 3]}
        res, mgr = run(ru, sensitivity, exp_cfg={n2: 0, n3: 4, n4: 0})
        assert res == {n2: 0, n3: 4, n4: 0}

        # reduce each ru target one at a time and check it's taken into account
        ru = ResourceUtilization(activation_memory=159, weights_memory=255, total_memory=415)
        run(ru, sensitivity, exp_cfg={n2: 1, n3: 4, n4: 0})

        ru = ResourceUtilization(activation_memory=160, weights_memory=254, total_memory=415)
        run(ru, sensitivity, exp_cfg={n2: 0, n3: 4, n4: 2})

        # n2 qcs: [a8, a2, a4]
        # n3 qcs: [a4w4, a4w8, a4w2, a8w4, a8w8, a8w2]
        # n4 qcs: [w12, w4, w8]
        ru = ResourceUtilization(activation_memory=160, weights_memory=255, total_memory=414)
        run(ru, sensitivity, exp_cfg={n2: 1, n3: 4, n4: 0})

        # min ru
        ru = ResourceUtilization(activation_memory=102.25, weights_memory=81.5, total_memory=183.75)
        res, mgr = run(ru, sensitivity, exp_cfg={n2: 1, n3: 2, n4: 1})
        assert res == mgr.min_ru_config

    def test_compute_ru_for_replaced_config(self, fw_info_mock, fw_impl_mock):
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=True, a_mp=True)
        ru = ResourceUtilization(weights_memory=100, activation_memory=200, total_memory=300)
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ru, mp_config=Mock())

        cfg = mgr.copy_config_with_replacement(mgr.min_ru_config, n3, 4)
        # make sure original cfg was not changed in place
        assert list(mgr.min_ru_config.values()) != list(cfg.values())
        assert cfg[n3] == 4
        ru_res = mgr.compute_resource_utilization_for_config(cfg)
        assert ru_res == ResourceUtilization(weights_memory=113, activation_memory=866/8, total_memory=221.25)

    @pytest.mark.parametrize('w_mp, a_mp, target_ru, exp_virtual', [
        (True, True, ResourceUtilization(activation_memory=1, weights_memory=2, total_memory=3), False),
        (True, True, ResourceUtilization(bops=1), True),
        (True, False, ResourceUtilization(bops=1), False),
        (False, True, ResourceUtilization(bops=1), False)
    ])
    def test_bops_no_bops_high_level_flow(self, fw_info_mock, fw_impl_mock, mocker, w_mp, a_mp, target_ru, exp_virtual):
        """ Tests that mp manager is instantiated correctly w.r.t virtual graph, and search method
            returns config w.r.t to the original graph in both cases. """
        g, _ = build_graph(fw_info_mock, w_mp=w_mp, a_mp=a_mp)

        substitute_mock = mocker.patch('model_compression_toolkit.core.common.mixed_precision.'
                                       'mixed_precision_search_manager.substitute')
        copy_mock = mocker.patch('model_compression_toolkit.core.common.mixed_precision.'
                                 'mixed_precision_search_manager.copy.deepcopy')
        compute_ru_mock = mocker.patch.object(MixedPrecisionRUHelper, 'compute_utilization')
        ru_helper_spy = mocker.patch('model_compression_toolkit.core.common.mixed_precision.'
                                     'mixed_precision_search_manager.MixedPrecisionRUHelper',
                                     Mock(wraps=MixedPrecisionRUHelper))
        cfg_recon_helper = mocker.patch('model_compression_toolkit.core.common.mixed_precision.'
                                        'mixed_precision_search_manager.ConfigReconstructionHelper',
                                        Mock(wraps=ConfigReconstructionHelper))
        reconstructed_configs = [Mock(), Mock()]
        recon_cfg_mock = mocker.patch.object(ConfigReconstructionHelper, 'reconstruct_full_configuration',
                                             Mock(side_effect=reconstructed_configs))
        prepare_and_run_mock = mocker.patch.object(MixedPrecisionSearchManager, '_prepare_and_run_solver')

        virt_sub_mock = Mock()
        fw_impl_mock.get_substitutions_virtual_weights_activation_coupling = virt_sub_mock

        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=target_ru, mp_config=Mock())
        res = mgr.search()
        # ru should always be computed on the original graph
        ru_helper_spy.assert_called_with(g, fw_impl_mock)
        if exp_virtual:
            substitute_mock.assert_called_with(copy_mock.return_value, virt_sub_mock.return_value)
            assert mgr.mp_graph is substitute_mock.return_value
            assert mgr.original_graph is g
            assert mgr.using_virtual_graph is True
            cfg_recon_helper.assert_called_with(g)
            assert mgr.config_reconstructor is not None
            # reconstruct_full_configuration should be called twice, in ctor for min_ru computation and on final config
            assert recon_cfg_mock.call_args_list == [call(mgr.min_ru_config), call(prepare_and_run_mock.return_value)]
            # min ru should be computed on reconstructed config
            assert compute_ru_mock.call_args_list[0] == call(mgr.ru_targets, reconstructed_configs[0])
            assert mgr.min_ru == compute_ru_mock.return_value
            assert res == reconstructed_configs[1]
        else:
            substitute_mock.assert_not_called()
            assert mgr.using_virtual_graph is False
            assert mgr.mp_graph is g and mgr.original_graph is g
            assert mgr.config_reconstructor is None
            recon_cfg_mock.assert_not_called()
            assert res == prepare_and_run_mock.return_value

    def test_build_sensitivity_mapping(self, fw_info_mock, fw_impl_mock):
        """ Test build sensitivity metric for regular graph (non-virtual)
            - real graph with real quantization candidates (dummy node types and weights)
            - check correct configurations (from graph) are passed to mock sensitivity evaluator.
            - final sensitivity is built correctly from mocked computed metrics.
        """
        fw_info_mock.get_kernel_op_attribute = lambda nt: 'foo' if nt is DummyLayer else None

        a_conf = build_node('a_conf', qcs=[build_nbits_qc(nb) for nb in (4, 2)], layer_class=DummyLayer2)
        a_conf_w = build_node('a_conf_w', canonical_weights={'foo': np.ones(10)},
                              qcs=[build_nbits_qc(nb, w_attr={'foo': (8, True)}) for nb in (4, 16)])
        w_conf = build_node('w_conf', canonical_weights={'foo': np.ones(10)},
                            qcs=[build_nbits_qc(4, w_attr={'foo': (nb, True)}) for nb in (8, 16)])
        aw_conf = build_node('aw_conf', canonical_weights={'foo': np.ones(10)},
                             qcs=[build_nbits_qc(ab, w_attr={'foo': (wb, True)})
                                  for ab, wb in [(4, 8), (4, 4), (8, 8), (8, 4)]])
        g = Graph(name='g', input_nodes=[a_conf], output_nodes=[aw_conf], nodes=[a_conf_w, aw_conf],
                  edge_list=[Edge(a_conf, a_conf_w, 0, 0), Edge(a_conf_w, w_conf, 0, 0), Edge(w_conf, aw_conf, 0, 0)])

        se = Mock(spec_set=SensitivityEvaluation)

        def compute_metric_mock(mp_a_cfg, mp_w_cfg):
            a = list(mp_a_cfg.values())[0] if mp_a_cfg else 0
            w = list(mp_w_cfg.values())[0] if mp_w_cfg else 0
            return a + 0.1*w
        se.compute_metric = Mock(side_effect=compute_metric_mock)

        mp_config = MixedPrecisionQuantizationConfig(metric_normalization=MpMetricNormalization.NONE,
                                                     metric_epsilon=None)
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=se,
                                          target_resource_utilization=ResourceUtilization(total_memory=100),
                                          mp_config=mp_config)
        res = mgr._build_sensitivity_mapping()
        call_args = se.compute_metric.call_args_list
        for i in range(2):
            assert call_args[i] == call(mp_a_cfg={'a_conf': i}, mp_w_cfg={}), i
        for i in range(2):
            assert call_args[2+i] == call(mp_a_cfg={'a_conf_w': i}, mp_w_cfg={}), i
        for i in range(2):
            assert call_args[4+i] == call(mp_a_cfg={}, mp_w_cfg={'w_conf': i})
        for i in range(4):
            assert call_args[6+i] == call(mp_a_cfg={'aw_conf': i}, mp_w_cfg={'aw_conf': i}), i
        assert len(res) == 4
        assert np.allclose(res[a_conf], np.array([0, 1]))
        assert np.allclose(res[a_conf_w], np.array([0, 1]))
        assert np.allclose(res[w_conf], np.array([0, 0.1]))
        assert np.allclose(res[aw_conf], np.array([0, 1.1, 2.2, 3.3]))

    def test_build_sensitivity_mapping_virtual_graph(self, graph_mock, fw_impl_mock, mocker):
        """ Test build_sensitivity method for virtual graph. We only test apis integration:
            - mock virtual graph, config reconstructor and sensitivity evaluator.
            - reconstruct_separate_aw_configs is called with correct args
            - correct configurations are passed to sensitivity evaluator compute_metrics
            - sensitivity matrix is built correctly from compute_metrics results
        """
        vg_mock = Mock()
        v_nodes = [Mock(candidates_quantization_cfg=[Mock(), Mock(), Mock()]),
                   Mock(candidates_quantization_cfg=[Mock(), Mock()])]
        vg_mock.get_configurable_sorted_nodes = lambda *args: v_nodes
        mocker.patch.object(MixedPrecisionSearchManager, '_get_mp_graph', Mock(return_value=(vg_mock, True)))

        se_mock = Mock()
        se_mock.compute_metric = Mock(side_effect=list(range(5)))

        mocker.patch.object(MixedPrecisionRUHelper, 'compute_utilization')
        mocker.patch.object(ConfigReconstructionHelper, 'reconstruct_full_configuration')
        # configs returned by config reconstructor
        aw_configs = [({build_node('a'): 3}, {build_node('b'): 5}),
                      ({build_node('c'): 2}, {}),
                      ({}, {build_node('c'): 0}),
                      ({build_node('c'): 1}, {build_node('d'): 2, build_node('e'): 3}),
                      ({build_node('d'): 5, build_node('e'): 4}, {})]
        recon_aw_cfgs_mock = mocker.patch.object(ConfigReconstructionHelper, 'reconstruct_separate_aw_configs',
                                                 Mock(side_effect=aw_configs))
        mp_config = MixedPrecisionQuantizationConfig(metric_epsilon=None,
                                                     metric_normalization=MpMetricNormalization.NONE)
        mgr = MixedPrecisionSearchManager(graph_mock, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=se_mock,
                                          target_resource_utilization=ResourceUtilization(bops=100),
                                          mp_config=mp_config)
        res = mgr._build_sensitivity_mapping()
        # check that config reconstruction is called with correct arguments
        assert recon_aw_cfgs_mock.call_args_list == [call({v_nodes[0]: 0}),
                                                     call({v_nodes[0]: 1}),
                                                     call({v_nodes[0]: 2}),
                                                     call({v_nodes[1]: 0}),
                                                     call({v_nodes[1]: 1})]
        # check that compute metrics is called with correct arguments
        assert se_mock.compute_metric.call_args_list == [call(mp_a_cfg={'a': 3}, mp_w_cfg={'b': 5}),
                                                         call(mp_a_cfg={'c': 2}, mp_w_cfg={}),
                                                         call(mp_a_cfg={}, mp_w_cfg={'c': 0}),
                                                         call(mp_a_cfg={'c': 1}, mp_w_cfg={'d': 2, 'e': 3}),
                                                         call(mp_a_cfg={'d': 5, 'e': 4}, mp_w_cfg={})]
        # check final results
        assert len(res) == 2
        assert np.array_equal(res[v_nodes[0]], np.array([0, 1, 2]))
        assert np.array_equal(res[v_nodes[1]], np.array([3, 4]))

    @pytest.mark.parametrize('norm, eps, max_thresh, exp1, exp2', [
        (MpMetricNormalization.NONE, None, 10, [1, 2, 3, 4], [1, 2, 3]),
        (MpMetricNormalization.NONE, None, 3.9, [.25, .5, .75, 1], [.25, .5, .75]),
        (MpMetricNormalization.NONE, 0, 10, [3, 3, 3, 4], [2, 2, 3]),
        (MpMetricNormalization.NONE, 0.1, 10, [3.1, 3.1, 3, 4], [2.1, 2, 3]),
        (MpMetricNormalization.MAXBIT, None, 10, [1 / 3, 2 / 3, 1, 4 / 3], [1 / 2, 1, 3 / 2]),
        (MpMetricNormalization.MAXBIT, 0, 10, [1, 1, 1, 4 / 3], [1, 1, 3 / 2]),
        (MpMetricNormalization.MINBIT, None, 10, [.5, 1, 1.5, 2], [1/3, 2/3, 1]),
        (MpMetricNormalization.MINBIT, 0.1, 10, [1.6, 1.6, 1.5, 2], [2 / 3 + .1, 2 / 3, 1]),
        (MpMetricNormalization.MINBIT, 0.1, 2, [.8, .8, .75, 1], [1 / 3 + .05, 1 / 3, .5])
    ])
    def test_build_sensitivity_mapping_params(self, fw_impl_mock, fw_info_mock, norm, eps, max_thresh, exp1, exp2):
        """ Tests sensitivity normalization method, epsilon and threshold. """
        fw_info_mock.get_kernel_op_attribute = lambda nt: None

        ph = build_node('ph', qcs=[build_nbits_qc()])
        n1 = build_node('n1', qcs=[build_nbits_qc(nb) for nb in (4, 2, 16, 8)])
        n2 = build_node('n2', qcs=[build_nbits_qc(nb) for nb in (4, 8, 2)])
        g = Graph(name='g', input_nodes=[ph], nodes=[n1], output_nodes=[n2],
                  edge_list=[Edge(ph, n1, 0, 0), Edge(n1, n2, 0, 0)])

        se = Mock(spec_set=SensitivityEvaluation)
        se.compute_metric = lambda mp_a_cfg, mp_w_cfg: list(mp_a_cfg.values())[0] + 1

        mp_config = MixedPrecisionQuantizationConfig(metric_normalization=norm,
                                                     metric_epsilon=eps,
                                                     metric_normalization_threshold=max_thresh)
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=se,
                                          target_resource_utilization=ResourceUtilization(activation_memory=100),
                                          mp_config=mp_config)
        mgr._finalize_distance_metric = Mock(wraps=mgr._finalize_distance_metric)

        res = mgr._build_sensitivity_mapping()
        assert set(res.keys()) == {n1, n2}
        assert np.allclose(res[n1], np.array(exp1))
        assert np.allclose(res[n2], np.array(exp2))
        mgr._finalize_distance_metric.assert_called_with(res)

    def _assert_dict_allclose(self, res, exp_res, sort_axis=None):
        assert len(exp_res) == len(res)
        for k in exp_res:
            if sort_axis is None:
                assert np.allclose(res[k], exp_res[k]), k
            else:
                assert np.allclose(np.sort(res[k], axis=sort_axis), np.sort(exp_res[k], axis=sort_axis)), k

    def _run_search_test(self, g, ru, sensitivity, exp_cfg, fw_impl):
        mgr = MixedPrecisionSearchManager(g, fw_impl=fw_impl, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ru, mp_config=Mock())
        mgr._build_sensitivity_mapping = Mock(return_value=sensitivity)
        res = mgr.search()
        assert res == exp_cfg
        return res, mgr


class TestConfigHelper:
    class AWLayer:
        pass

    class ALayer:
        pass

    class VALayer:
        pass

    kernel_attr = 'im_kernel'

    @pytest.fixture(autouse=True)
    def setup(self, fw_info_mock):
        fw_info_mock.get_kernel_op_attribute = \
            lambda nt: self.kernel_attr if nt == self.AWLayer else None
        set_fw_info(fw_info_mock)

    @staticmethod
    def build_aw_node(abits, wbits, name='aw', layer_cls=AWLayer, w_attr=kernel_attr):
        qcs = []
        for abit in abits:
            for wbit in wbits:
                qcs.append(build_nbits_qc(abit, True, w_attr={w_attr: (wbit, True)}))
        return build_node(name, canonical_weights={w_attr: np.ones(50)}, qcs=qcs, layer_class=layer_cls)

    @staticmethod
    def build_a_node(abits, name='a', layer_cls=ALayer):
        return build_node(name, qcs=[build_nbits_qc(abit) for abit in abits], layer_class=layer_cls)

    @pytest.mark.parametrize('n_params, ind', [
        ({'abits': (5, 3, 7)}, 1),
        ({'abits': (4, 8, 16), 'wbits': (2, 6, 10)}, 8),
        ({'abits': (4, 8, 16), 'wbits': (2, 6, 10)}, 5),
        ({'abits': (5,), 'wbits': (2, 4, 8)}, 2),
    ])
    def test_retrieve_a_candidates_regular_node(self, graph_mock, n_params, ind):
        n = self.build_aw_node(**n_params) if 'wbits' in n_params else self.build_a_node(**n_params)
        graph_mock.nodes = [n]
        helper = ConfigReconstructionHelper(graph_mock)
        ret_n, ret_inds = helper._retrieve_matching_orig_a_candidates(copy.deepcopy(n), ind)
        assert ret_n is n
        assert ret_inds == [ind]

    @pytest.mark.parametrize('abits, wbits, vind', [
        ((4, 8, 16), (2, 6, 10, 12), 0),
        ((4, 8, 16), (2, 6, 10, 12), 2),
        ((4,), (2, 6, 10), 0)
    ])
    def test_retrieve_a_candidates_virt_split_node(self, graph_mock, abits, wbits, vind):
        orig_n = self.build_aw_node(abits=abits, wbits=wbits)
        graph_mock.nodes = [orig_n]
        helper = ConfigReconstructionHelper(graph_mock)
        van = VirtualSplitActivationNode(copy.deepcopy(orig_n), self.VALayer, {})
        ret_n, ret_inds = helper._retrieve_matching_orig_a_candidates(van, vind)
        assert ret_n is orig_n
        assert len(ret_inds) == len(wbits)
        self._validate_activation(orig_n, ret_inds, van, vind)

        vwn = VirtualSplitWeightsNode(orig_n, self.kernel_attr)
        assert helper._retrieve_matching_orig_a_candidates(vwn, vind) == (None, None)

    @pytest.mark.parametrize('orig_a_n, build_va, orig_w_n, v_ind', [
        ((build_a_node, {'abits': (2, 4, 8)}), False, (build_aw_node, {'abits': (5, 7, 3), 'wbits': (3, 6)}), 5),
        ((build_a_node, {'abits': (2,)}), False, (build_aw_node, {'abits': (5, 7), 'wbits': (3, 4, 6)}), 2),
        ((build_aw_node, {'abits': (2, 4, 6), 'wbits': (3, 5, 7, 9)}), True, (build_aw_node, {'abits': (8, 10), 'wbits': (11, 12)}), 5),
        ((build_aw_node, {'abits': (2, 4, 6), 'wbits': (3, 5, 7, 9)}), True, (build_aw_node, {'abits': (8, 10), 'wbits': (11, 12)}), 3),
        ((build_aw_node, {'abits': (2,), 'wbits': (3,)}), True, (build_aw_node, {'abits': (8,), 'wbits': (11,)}), 0)
    ])
    def test_retrieve_a_candidates_virt_aw_node(self, graph_mock, orig_a_n, build_va, orig_w_n, v_ind):
        orig_a_n = orig_a_n[0](**orig_a_n[1])
        orig_w_n = orig_w_n[0](**orig_w_n[1])
        graph_mock.nodes = [orig_a_n]
        a_n = copy.deepcopy(orig_a_n)
        if build_va:
            a_n = VirtualSplitActivationNode(a_n, self.VALayer, {})
        vaw_n = VirtualActivationWeightsNode(act_node=a_n,
                                             weights_node=VirtualSplitWeightsNode(orig_w_n, self.kernel_attr))
        helper = ConfigReconstructionHelper(graph_mock)
        ret_n, ret_inds = helper._retrieve_matching_orig_a_candidates(vaw_n, v_ind)
        assert ret_n is orig_a_n
        self._validate_activation(orig_a_n, ret_inds, vaw_n, v_ind)

    @pytest.mark.parametrize('n_params, ind', [
        ({'abits': (4, 8, 16), 'wbits': (2, 6, 10)}, 8),
        ({'abits': (4, 8, 16), 'wbits': (2, 6, 10)}, 1),
        ({'abits': (5,), 'wbits': (2, 4, 8)}, 1),
        ({'abits': (4, 8, 16), 'wbits': (5,)}, 2)
    ])
    def test_retrieve_w_candidates_regular_node(self, graph_mock, n_params, ind):
        n = self.build_aw_node(**n_params)
        graph_mock.nodes = [n]
        helper = ConfigReconstructionHelper(graph_mock)
        ret_n, ret_inds = helper._retrieve_matching_orig_w_candidates(copy.deepcopy(n), ind)
        assert ret_n is n
        assert ret_inds == [ind]

    def test_retrieve_w_candidates_activation_node(self, graph_mock):
        n = self.build_a_node(abits=(5, 3, 7))
        graph_mock.nodes = [n]
        helper = ConfigReconstructionHelper(graph_mock)
        assert helper._retrieve_matching_orig_w_candidates(n, 0) == (None, None)

    @pytest.mark.parametrize('abits, wbits, vind', [
        ((4, 8, 16), (2, 6, 10, 12), 0),
        ((4, 8, 16), (2, 6, 10, 12), 2),
        ((4,), (2, 6, 10), 0),
        ((2, 6, 10), (4,), 0)
    ])
    def test_retrieve_w_candidates_virt_split_node(self, graph_mock, abits, wbits, vind):
        orig_n = self.build_aw_node(abits=abits, wbits=wbits)
        graph_mock.nodes = [orig_n]
        vw_n = VirtualSplitWeightsNode(copy.deepcopy(orig_n), self.kernel_attr)
        helper = ConfigReconstructionHelper(graph_mock)
        ret_n, ret_inds = helper._retrieve_matching_orig_w_candidates(vw_n, vind)
        assert ret_n is orig_n
        self._validate_weights(orig_n, ret_inds, vw_n, vind)

        va_n = VirtualSplitActivationNode(copy.deepcopy(orig_n), self.VALayer, {})
        assert helper._retrieve_matching_orig_w_candidates(va_n, vind) == (None, None)

    @pytest.mark.parametrize('orig_a_n, build_va, orig_w_n, v_ind', [
        ({'abits': (2, 4, 8, 16)}, False, {'abits': (5, 7), 'wbits': (3, 6, 9)}, 11),
        ({'abits': (2, 4, 8, 16)}, True, {'abits': (5, 7), 'wbits': (3, 6, 9)}, 7),
        ({'abits': (2, 4, 8, 16)}, True, {'abits': (5, 7), 'wbits': (3,)}, 2),
        ({'abits': (2,)}, False, {'abits': (5, 7), 'wbits': (3, 4, 6)}, 2),
        ({'abits': (2,)}, True, {'abits': (8,), 'wbits': (11,)}, 0)
    ])
    def test_retrieve_w_candidates_virt_aw_node(self, graph_mock, orig_a_n, build_va, orig_w_n, v_ind):
        orig_a_n = self.build_a_node(**orig_a_n)
        orig_w_n = self.build_aw_node(**orig_w_n)
        graph_mock.nodes = [orig_a_n]
        a_n = copy.deepcopy(orig_a_n)
        if build_va:
            a_n = VirtualSplitActivationNode(a_n, self.VALayer, {})
        vaw_n = VirtualActivationWeightsNode(act_node=a_n,
                                             weights_node=VirtualSplitWeightsNode(orig_w_n, self.kernel_attr))
        helper = ConfigReconstructionHelper(graph_mock)
        ret_n, ret_inds = helper._retrieve_matching_orig_a_candidates(vaw_n, v_ind)
        assert ret_n is orig_a_n
        self._validate_activation(orig_a_n, ret_inds, vaw_n, v_ind)

    @pytest.mark.parametrize('build_va, v_ind', [(False, 5)]) #[(True, 8), (False, 5)])
    def test_retrieve_w_candidates_virt_aw_node_multiple_weights(self, graph_mock, build_va, v_ind):
        """ A node contains non-kernel non-configurable weights, w node contains additional non-configurable weight,
            some of weight attrs are identical. Configs should be retrieved by configurable weight of w_node. """
        orig_a_node = build_node('a', canonical_weights={'foo': np.ones(2), 'bar': np.ones(3), 1: np.ones(4)},
                                 qcs=[build_nbits_qc(ab, True, w_attr={'foo': (4, True), 'bar': (5, True)},
                                                     pos_attr=(6, True, [1])) for ab in (2, 3, 7)],
                                 layer_class=self.ALayer)
        orig_w_node = build_node('a',
                                 canonical_weights={'bar': np.ones(2), self.kernel_attr: np.ones(3), 1: np.ones(4)},
                                 qcs=[build_nbits_qc(ab, True, w_attr={'bar': (4, True), self.kernel_attr: (wb, True)},
                                                     pos_attr=(6, True, [1])) for ab in (2, 3) for wb in (9, 10, 11)],
                                 layer_class=self.AWLayer)
        graph_mock.nodes = [orig_a_node, orig_w_node]
        a_node = copy.deepcopy(orig_a_node)
        if build_va:
            a_node = VirtualSplitActivationNode(a_node, self.ALayer, {})
        vaw = VirtualActivationWeightsNode(act_node=a_node,
                                           weights_node=VirtualSplitWeightsNode(copy.deepcopy(orig_w_node), self.kernel_attr))
        helper = ConfigReconstructionHelper(graph_mock)
        ret_n, ret_inds = helper._retrieve_matching_orig_w_candidates(vaw, v_ind)
        assert ret_n is orig_w_node
        self._validate_weights(orig_w_node, ret_inds, vaw, v_ind, attrs=[self.kernel_attr])

    def test_reconstruct_separate_aw_configs_regular_nodes(self, graph_mock):
        mpa = self.build_a_node(name='mpa', abits=(4, 8, 16))
        mpa_mpw = self.build_aw_node(name='mpa_mpw', abits=(2, 3, 4), wbits=(5, 6))
        spa_mpw = self.build_aw_node(name='spa_mpw', abits=(2,), wbits=(5, 6, 7))
        mpa_spw = self.build_aw_node(name='mpa_spw', abits=(2, 3, 4), wbits=(5,))
        graph_mock.nodes = [mpa, mpa_mpw, spa_mpw, mpa_spw]

        helper = ConfigReconstructionHelper(graph_mock)
        v_cfg = {mpa: 1, mpa_mpw: 4, spa_mpw: 2, mpa_spw: 1}
        a_cfg, w_cfg = helper.reconstruct_separate_aw_configs(v_cfg, include_non_configurable=False)
        assert a_cfg == {n: v_cfg[n] for n in (mpa, mpa_mpw, mpa_spw)}
        assert w_cfg == {n: v_cfg[n] for n in (mpa_mpw, spa_mpw)}

        a_cfg, w_cfg = helper.reconstruct_separate_aw_configs(v_cfg, include_non_configurable=True)
        assert a_cfg == {n: v_cfg[n] for n in (mpa, mpa_mpw, spa_mpw, mpa_spw)}
        assert w_cfg == {n: v_cfg[n] for n in (mpa_mpw, spa_mpw, mpa_spw)}

    @pytest.mark.parametrize('incl_non_conf', [True, False])
    def test_reconstruct_separate_aw_configs_virtual_nodes(self, graph_mock, incl_non_conf):
        spa = self.build_a_node(name='spa', abits=(4,))
        mpa_mpw = self.build_aw_node(name='mpa_mpw', abits=(2, 3, 4), wbits=(5, 6, 7))
        mpa_spw = self.build_aw_node(name='mpa_spw', abits=(7, 8, 9), wbits=(5,))
        mpa_mpw2 = self.build_aw_node(name='mpa_mpw2', abits=(6, 7), wbits=(3, 4, 5))
        graph_mock.nodes = [spa, mpa_mpw, mpa_spw, mpa_mpw2]
        vaw1 = VirtualActivationWeightsNode(copy.deepcopy(spa),
                                            VirtualSplitWeightsNode(copy.deepcopy(mpa_mpw), self.kernel_attr))
        va1 = VirtualSplitActivationNode(copy.deepcopy(mpa_spw), self.ALayer, {})
        vw = VirtualSplitWeightsNode(copy.deepcopy(mpa_mpw2), self.kernel_attr)
        vaw2 = VirtualActivationWeightsNode(VirtualSplitActivationNode(copy.deepcopy(mpa_mpw2), self.ALayer, {}),
                                            VirtualSplitWeightsNode(copy.deepcopy(mpa_spw), self.kernel_attr))

        v_cfg = {vaw1: 2, va1: 1, vw: 2, vaw2: 1}

        def validate_activation(n, vn, ret_cfg):
            assert self._get_activation_cfg(n, ret_cfg[n]) == self._get_activation_cfg(vn, v_cfg[vn])

        def validate_weights(n, vn, ret_cfg):
            assert self._get_weights_cfg(n, ret_cfg[n]) == self._get_weights_cfg(vn, v_cfg[vn])

        helper = ConfigReconstructionHelper(graph_mock)
        a_cfg, w_cfg = helper.reconstruct_separate_aw_configs(v_cfg, include_non_configurable=incl_non_conf)
        if incl_non_conf:
            assert set(a_cfg.keys()) == {spa, mpa_spw, mpa_mpw2}
            assert set(w_cfg.keys()) == {mpa_spw, mpa_mpw, mpa_mpw2}
        else:
            assert set(a_cfg.keys()) == {mpa_spw, mpa_mpw2}
            assert set(w_cfg.keys()) == {mpa_mpw, mpa_mpw2}
        validate_activation(mpa_spw, va1, a_cfg)
        validate_activation(mpa_mpw2, vaw2, a_cfg)
        validate_weights(mpa_mpw, vaw1, w_cfg)
        validate_weights(mpa_mpw2, vw, w_cfg)

    @pytest.mark.parametrize('incl_non_conf', [True, False])
    def test_reconstruct_full_configuration(self, graph_mock, incl_non_conf):
        # orig nodes for virtual nodes
        spa = self.build_a_node(name='spa', abits=(4,))
        mpa_mpw = self.build_aw_node(name='mpa_mpw', abits=(2, 3, 4), wbits=(5, 6, 7))
        mpa_spw = self.build_aw_node(name='mpa_spw', abits=(7, 8, 9), wbits=(5,))
        spa_mpw = self.build_aw_node(name='spa_mpw', abits=(6,), wbits=(3, 4, 5))
        # orig nodes as is
        mpa_mpw2 = self.build_aw_node(name='mpa_mpw2', abits=(4, 5, 6), wbits=(2, 6, 8))
        mpa_spw2 = self.build_aw_node(name='mpa_spw2', abits=(4, 5, 6), wbits=(3,))
        spa_mpw2 = self.build_aw_node(name='spa_spw2', abits=(5,), wbits=(2, 6, 8))

        graph_mock.nodes = [spa, mpa_mpw, mpa_spw, spa_mpw, mpa_mpw2, mpa_spw2, spa_mpw2]
        vaw1 = VirtualActivationWeightsNode(copy.deepcopy(spa),
                                            VirtualSplitWeightsNode(copy.deepcopy(mpa_mpw), self.kernel_attr))
        vaw2 = VirtualActivationWeightsNode(VirtualSplitActivationNode(copy.deepcopy(mpa_mpw), self.ALayer, {}),
                                            VirtualSplitWeightsNode(copy.deepcopy(mpa_spw), self.kernel_attr))
        va = VirtualSplitActivationNode(copy.deepcopy(mpa_spw), self.ALayer, {})
        vw = VirtualSplitWeightsNode(copy.deepcopy(spa_mpw), self.kernel_attr)

        v_cfg = {vaw1: 2, vaw2: 1, va: 2, vw: 2, mpa_mpw2: 7, mpa_spw2: 1, spa_mpw2: 2}

        def validate_activation(n, vn, ret_cfg):
            assert self._get_activation_cfg(n, ret_cfg[n]) == self._get_activation_cfg(vn, v_cfg[vn]), n

        def validate_weights(n, vn, ret_cfg):
            assert self._get_weights_cfg(n, ret_cfg[n]) == self._get_weights_cfg(vn, v_cfg[vn]), n

        helper = ConfigReconstructionHelper(graph_mock)
        ret_cfg = helper.reconstruct_full_configuration(v_cfg, include_non_configurable=incl_non_conf)

        exp_keys = {mpa_mpw, mpa_spw, spa_mpw, mpa_mpw2, spa_mpw2, mpa_spw2}
        if incl_non_conf:
            exp_keys.add(spa)
            assert ret_cfg[spa] == 0
        assert set(ret_cfg.keys()) == exp_keys
        validate_activation(mpa_mpw, vaw2, ret_cfg)
        validate_weights(mpa_mpw, vaw1, ret_cfg)
        validate_activation(mpa_spw, va, ret_cfg)
        validate_weights(spa_mpw, vw, ret_cfg)
        for n in (mpa_mpw2, spa_mpw2, mpa_spw2):
            assert v_cfg[n] == ret_cfg[n], n

    def _validate_activation(self, orig_n, ret_inds, vn, vind):
        """ Validates that all returned candidate indices of the original node match activation config of the virtual
            candidate, and all matching indices have been retrieved. """
        for i, _ in enumerate(orig_n.candidates_quantization_cfg):
            if i in ret_inds:
                assert self._get_activation_cfg(orig_n, i) == self._get_activation_cfg(vn, vind)
            else:
                assert (self._get_activation_cfg(orig_n, i).activation_n_bits !=
                        self._get_activation_cfg(vn, vind).activation_n_bits)

    def _validate_weights(self, orig_n, ret_inds, vn, vind, attrs=None):
        """ Validates that all returned candidate indices of the original node match eights config of the virtual
            candidate, and all matching indices have been retrieved. """
        for i, _ in enumerate(orig_n.candidates_quantization_cfg):
            orig_cfg = self._get_weights_cfg(orig_n, i)
            v_cfg = self._get_weights_cfg(vn, vind)
            if i in ret_inds:
                if attrs:
                    assert all(orig_cfg.get_attr_config(attr) == v_cfg.get_attr_config(attr) for attr in attrs)
                else:
                    assert orig_cfg == v_cfg
            else:
                attrs = attrs or orig_n.weights.keys()
                assert any(orig_cfg.get_attr_config(attr).weights_n_bits != v_cfg.get_attr_config(attr).weights_n_bits
                           for attr in attrs)

    @staticmethod
    def _get_activation_cfg(n, ind):
        return n.candidates_quantization_cfg[ind].activation_quantization_cfg

    @staticmethod
    def _get_weights_cfg(n, ind):
        return n.candidates_quantization_cfg[ind].weights_quantization_cfg
