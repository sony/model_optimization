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
import pytest
from unittest.mock import Mock

import numpy as np

from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.core.common.framework_info import DEFAULT_KERNEL_ATTRIBUTES
from model_compression_toolkit.core.common.graph.edge import Edge
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_ru_helper import MixedPrecisionRUHelper
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_search_manager import \
    MixedPrecisionSearchManager, ConfigReconstructionHelper
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc


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
    n1 = build_node('n1', qcs=[build_nbits_qc(a_nbits=8)], output_shape=(None, 1, 2, 3))

    abits2 = [8, 2, 4] if a_mp else [4]
    n2 = build_node('n2', qcs=[build_nbits_qc(a_nbits=ab) for ab in abits2], output_shape=(None, 4, 5, 6))

    abit3 = [4, 8] if a_mp else [8]
    wbit3 = [4, 8, 2] if w_mp else [4]
    n3 = build_node('n3', layer_class=DummyLayer1, output_shape=(None, 5, 8),
                    canonical_weights={'foo': np.ones((3, 14))},
                    qcs=[build_nbits_qc(a_nbits=ab, w_attr={'foo': (wb, True)}) for ab in abit3 for wb in wbit3])

    wbit4 = [12, 4, 8] if w_mp else [8]
    n4 = build_node('n4', layer_class=DummyLayer2, output_shape=(None, 13, 21),
                    canonical_weights={'bar': np.ones((2, 71))},
                    qcs=[build_nbits_qc(a_nbits=2, w_attr={'bar': (wb, True)}) for wb in wbit4])
    n5 = build_node('n5', qcs=[build_nbits_qc(a_nbits=8)], output_shape=(None, 34, 1))

    op_kernels = {DummyLayer1: ['foo'], DummyLayer2: ['bar']}
    fw_info_mock.get_kernel_op_attributes = lambda nt: op_kernels.get(nt, DEFAULT_KERNEL_ATTRIBUTES)
    fw_info_mock.is_kernel_op = lambda nt: nt in op_kernels

    g = Graph('g', input_nodes=[n1], nodes=[n2, n3, n4], output_nodes=[n5],
              edge_list=[Edge(n1, n2, 0, 0), Edge(n2, n3, 0, 0), Edge(n3, n4, 0, 0), Edge(n4, n5, 0, 0)],
              fw_info=fw_info_mock)
    return g, [n1, n2, n3, n4, n5]


class TestMixedPrecisionSearchManager:
    """ MP search manager tests.
        TODO: Sensitivity computation is not tested.
              BOPS: only logical flow is tested.
    """
    def test_prepare_weights_ru_for_lp(self, fw_info_mock, fw_impl_mock):
        """ Tests ru related setup and methods for weights target. """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=True, a_mp=False)
        ru = ResourceUtilization(weights_memory=100)
        mgr = MixedPrecisionSearchManager(g, fw_info=fw_info_mock, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(), target_resource_utilization=ru)
        assert mgr.min_ru_config == {n3: 2, n4: 1}
        assert mgr.max_ru_config == {n3: 1, n4: 0}
        assert mgr.min_ru == {RUTarget.WEIGHTS: 3 * 14 * 2 / 8 + 2 * 71 * 4 / 8}

        rel_ru = mgr._compute_relative_ru_matrices()
        self._assert_dict_allclose(rel_ru, {RUTarget.WEIGHTS: np.array([2*42, 6*42, 0, 8*142, 0, 4*142])[:, None]/8})
        rel_constraint = mgr._get_relative_ru_constraint_per_mem_element()
        self._assert_dict_allclose(rel_constraint, {RUTarget.WEIGHTS: np.array([[100 - 81.5]])})

    def test_prepare_activation_ru_for_lp(self, fw_info_mock, fw_impl_mock):
        """ Tests ru related setup and methods for activation target. """
        g, [n1, n2, n3, n4, n5] = build_graph(fw_info_mock, w_mp=False, a_mp=True)
        ru = ResourceUtilization(activation_memory=150)
        mgr = MixedPrecisionSearchManager(g, fw_info=fw_info_mock, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(), target_resource_utilization=ru)
        # 6 x [8], 120 x [8, 2, 4], 40 x [4, 8], 273 x [2], 34 x [8]
        assert mgr.min_ru_config == {n2: 1, n3: 0}
        assert mgr.max_ru_config == {n2: 0, n3: 1}
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
        mgr = MixedPrecisionSearchManager(g, fw_info=fw_info_mock, fw_impl=fw_impl_mock,
                                          sensitivity_evaluator=Mock(), target_resource_utilization=ru)
        assert mgr.min_ru_config == {n2: 1, n3: 2, n4: 1}
        assert mgr.max_ru_config == {n2: 0, n3: 4, n4: 0}

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
        assert res == mgr.max_ru_config

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
        assert res == mgr.max_ru_config

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
        mgr = MixedPrecisionSearchManager(g, fw_info=g.fw_info, fw_impl=fw_impl_mock, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ru)

        cfg = mgr.copy_config_with_replacement(mgr.min_ru_config, n3, 4)
        # make sure original cfg was not changed in place
        assert list(mgr.min_ru_config.values()) != list(cfg.values())
        assert cfg[n3] == 4
        ru_res = mgr.compute_resource_utilization_for_config(cfg)
        assert ru_res == ResourceUtilization(weights_memory=113, activation_memory=866/8, total_memory=221.25)

    def test_bops_no_bops_flow(self, fw_info_mock, fw_impl_mock, mocker):
        g, _ = build_graph(fw_info_mock, w_mp=True, a_mp=True)

        substitute_mock = mocker.patch('model_compression_toolkit.core.common.mixed_precision.'
                                       'mixed_precision_search_manager.substitute')
        copy_mock = mocker.patch('model_compression_toolkit.core.common.mixed_precision.'
                                 'mixed_precision_search_manager.copy.deepcopy')
        mocker.patch.object(MixedPrecisionRUHelper, 'compute_utilization')

        recon_cfg_mock = mocker.patch.object(ConfigReconstructionHelper, 'reconstruct_config_from_virtual_graph')
        mocker.patch.object(MixedPrecisionSearchManager, '_prepare_and_run_solver')

        virt_sub_mock = Mock()
        fw_impl_mock.get_substitutions_virtual_weights_activation_coupling = virt_sub_mock

        # no bops
        ru_no_bops = ResourceUtilization(activation_memory=1, weights_memory=2, total_memory=3)
        mgr = MixedPrecisionSearchManager(g, fw_info=fw_info_mock, fw_impl=fw_impl_mock, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ru_no_bops)
        mgr.search()

        substitute_mock.assert_not_called()
        assert mgr.using_virtual_graph is False
        assert mgr.mp_graph is g and mgr.original_graph is g
        recon_cfg_mock.assert_not_called()

        # with bops
        ru_bops = ResourceUtilization(activation_memory=1, bops=2)
        mgr = MixedPrecisionSearchManager(g, fw_info=g.fw_info, fw_impl=fw_impl_mock, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ru_bops)
        res = mgr.search()
        substitute_mock.assert_called_with(copy_mock.return_value, virt_sub_mock.return_value)
        assert mgr.mp_graph is substitute_mock.return_value
        assert mgr.original_graph is g
        assert mgr.using_virtual_graph is True
        recon_cfg_mock.assert_called()
        assert res == recon_cfg_mock.return_value

    def _assert_dict_allclose(self, res, exp_res, sort_axis=None):
        assert len(exp_res) == len(res)
        for k in exp_res:
            if sort_axis is None:
                assert np.allclose(res[k], exp_res[k]), k
            else:
                assert np.allclose(np.sort(res[k], axis=sort_axis), np.sort(exp_res[k], axis=sort_axis)), k

    def _run_search_test(self, g, ru, sensitivity, exp_cfg, fw_impl):
        mgr = MixedPrecisionSearchManager(g, fw_info=g.fw_info, fw_impl=fw_impl, sensitivity_evaluator=Mock(),
                                          target_resource_utilization=ru)
        mgr._build_sensitivity_mapping = Mock(return_value=sensitivity)
        res = mgr.search()
        assert res == exp_cfg
        return res, mgr
