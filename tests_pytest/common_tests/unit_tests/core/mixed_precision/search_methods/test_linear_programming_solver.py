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
import numpy as np
import pytest

from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    MixedPrecisionIntegerLPSolver


class TestMixedPrecisionIntegerLPSolver:
    @pytest.mark.parametrize('ru_target', [RUTarget.WEIGHTS, RUTarget.BOPS])
    def test_weights_or_bops_constraint(self, ru_target):
        """ Test ru targets with scalar constraint (weights, bops). """
        sensitivity = {'n1': [0.1, 0.4, 0.3], 'n2': [0.35, 0.3], 'n3': [0.7, 0.3, 0.8, 0.2]}
        ru = {ru_target: np.array([3, 2, 1] + [4, 4] + [5, 6, 7, 8])[:, None]}

        for c in [20, 15]:
            self._run_test(sensitivity, ru,  {ru_target: np.array([c])}, exp_res={'n1': 0, 'n2': 1, 'n3': 3})

        for c in [14.99, 13]:
            self._run_test(sensitivity, ru, {ru_target: np.array([c])}, exp_res={'n1': 0, 'n2': 1, 'n3': 1})

        for c in [12.99, 11]:
            self._run_test(sensitivity, ru, {ru_target: np.array([c])}, exp_res={'n1': 2, 'n2': 1, 'n3': 1})

        for c in [10.99, 10]:
            self._run_test(sensitivity, ru, {ru_target: np.array([c])}, exp_res={'n1': 2, 'n2': 1, 'n3': 0})

        with pytest.raises(RuntimeError, match='No solution was found for the LP problem'):
            self._run_test(sensitivity, ru, {ru_target: np.array([9.99])}, None)

    @pytest.mark.parametrize('ru_target', [RUTarget.ACTIVATION, RUTarget.TOTAL])
    def test_activation_or_total_constraint(self, ru_target):
        """ Test ru targets with multiple memory elements (cuts).
            Constraints for all cuts should be met in order for a solution to be selected. """
        sensitivity = {'n1': [0.1, 0.4, 0.3], 'n2': [0.35, 0.3], 'n3': [0.7, 0.3, 0.8, 0.2]}
        # Optimal candidates (lowest sensitivity) have the largest ru in some cut (so that they can be filtered out)
        # Worst candidates have a smaller ru in some cut than other candidates in some cut (so that with sufficiently
        # low constraint no other candidate meets the constraints for all cuts)
        ru = {ru_target: np.array([[3, 2, 1] + [4, 4] + [7, 6, 5, 8],
                                   [1, 2, 3] + [4, 4] + [8, 5, 6, 7],
                                   [5, 6, 7] + [4, 8] + [4, 2, 1, 8],
                                   [8, 7, 4] + [3, 2] + [6, 5, 4, 1]]).T}

        # optimal solution, tight constraint (ru==constraint per cut)
        ru_constraints = np.array([3+4+8, 1+4+7, 5+8+8, 8+2+1])
        self._run_test(sensitivity, ru,  {ru_target: ru_constraints}, exp_res={'n1': 0, 'n2': 1, 'n3': 3})

        # 3 cuts meet the constraint for the optimal solution, and only one (non-maximal) does not ->
        # optimal solution should not be selected (last cut is increased so that the second best solution fits).
        ru_constraints = np.array([3+4+8, 1+4+7-0.01, 5+8+8, 8+2+5])
        self._run_test(sensitivity, ru, {ru_target: ru_constraints}, exp_res={'n1': 0, 'n2': 1, 'n3': 1})

        # second best solution, tight constraints
        ru_constraints = np.array([3+4+6, 1+4+5, 5+8+2, 8+2+5])
        self._run_test(sensitivity, ru, {ru_target: ru_constraints}, exp_res={'n1': 0, 'n2': 1, 'n3': 1})

        # worst solution, tight constraints (no other candidates meet the constraints for all cuts)
        ru_constraints = np.array([2+4+5, 2+4+6, 6+4+1, 7+3+4])
        self._run_test(sensitivity, ru, {ru_target: ru_constraints}, exp_res={'n1': 1, 'n2': 0, 'n3': 2})

        # worst candidates - relax constraints as long as other candidates still don't meet the constraint for all cuts
        ru_constraints = np.array([100, 100, 6+4+1, 7+3+4])
        self._run_test(sensitivity, ru, {ru_target: ru_constraints}, exp_res={'n1': 1, 'n2': 0, 'n3': 2})

        # 2 pairs of candidates meet the constraint of the 3rd cut, select the one with lower sensitivity
        ru_constraints = np.array([100, 100, 14.9, 100])
        self._run_test(sensitivity, ru, {ru_target: ru_constraints}, exp_res={'n1': 0, 'n2': 0, 'n3': 1})

        # flip to next solution
        ru_constraints = np.array([100, 100, 15., 100])
        self._run_test(sensitivity, ru, {ru_target: ru_constraints}, exp_res={'n1': 0, 'n2': 1, 'n3': 1})

        # it's enough that one cut doesn't meet the constraint
        with pytest.raises(RuntimeError, match='No solution was found for the LP problem'):
            self._run_test(sensitivity, ru, {ru_target: np.array([11, 12-0.1, 11, 14])}, None)

    def test_all_ru_targets(self):
        """ Check that all ru targets are taken into account. """
        sensitivity = {'n1': [0.1, 0.3, 0.2], 'n2': [0.4, 0.3], 'n3': [0.4, 0.3, 0.5, 0.2]}
        # all layers and memory element have identical ru
        ru = {
             RUTarget.WEIGHTS: np.ones((9, 1)),
             RUTarget.ACTIVATION: 2*np.ones((9, 5)),
             RUTarget.TOTAL: 3*np.ones((9, 5)),
             RUTarget.BOPS: 4*np.ones((9, 1))
        }
        # tight constraint
        ru_constraints = {
            RUTarget.WEIGHTS: np.array([3]),
            RUTarget.ACTIVATION: 6*np.ones(5),
            RUTarget.TOTAL: 9*np.ones(5),
            RUTarget.BOPS: np.array([12])
        }

        # optimal solution
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 0, 'n2': 1, 'n3': 3})

        # increase weights ru for the optimal candidate of the 3rd layer
        ru[RUTarget.WEIGHTS][8, 0] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 0, 'n2': 1, 'n3': 1})

        # in addition, increase activation ru for one of the cuts of the current optimal candidate of the 3rd layer
        ru_constraints[RUTarget.ACTIVATION][6, 2] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 0, 'n2': 1, 'n3': 0})

        # in addition, increase total ru for one of the cuts of the optimal candidate of the 2nd layer
        ru[RUTarget.TOTAL][0, 4] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 2, 'n2': 1, 'n3': 0})

        # in addition, increase bops for the optimal candidate of 2nd layer above constraint
        ru[RUTarget.BOPS][4, 0] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 2, 'n2': 2, 'n3': 0})

    def _run_test(self, sensitivity, ru, ru_constraints, exp_res):
        solver = MixedPrecisionIntegerLPSolver(sensitivity, ru, ru_constraints)
        res = solver.run()
        assert res == exp_res
