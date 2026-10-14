# Copyright 2025 Sony Semiconductor Solutions, Inc. All rights reserved.
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
from types import SimpleNamespace

import numpy as np
import pulp
import pytest

from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget
from model_compression_toolkit.core.common.mixed_precision.search_methods.linear_programming import \
    MixedPrecisionIntegerLPSolver
from model_compression_toolkit.logger import Logger


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
        ru[RUTarget.ACTIVATION][6, 2] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 0, 'n2': 1, 'n3': 0})

        # in addition, increase total ru for one of the cuts of the optimal candidate of the 2nd layer
        ru[RUTarget.TOTAL][0, 4] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 2, 'n2': 1, 'n3': 0})

        # in addition, increase bops for the optimal candidate of 2nd layer above constraint
        ru[RUTarget.BOPS][4, 0] += 0.1
        self._run_test(sensitivity, ru, ru_constraints, {'n1': 2, 'n2': 0, 'n3': 0})

    def test_filter_all_finite_candidates(self, mocker):
        """Test that finite candidates are not modified or logged."""
        sensitivity = {'n1': [0.1, 0.2], 'n2': [0.3]}
        ru = {
            RUTarget.WEIGHTS: np.array([[10], [11], [12]]),
            RUTarget.ACTIVATION: np.array([[20, 21], [22, 23], [24, 25]]),
            RUTarget.TOTAL: np.array([[30, 31], [32, 33], [34, 35]]),
            RUTarget.BOPS: np.array([[40], [41], [42]]),
        }
        warning_mock = mocker.patch.object(Logger, 'warning')

        filtered_sensitivity, filtered_ru, candidate_index_mapping = \
            MixedPrecisionIntegerLPSolver._filter_non_finite_candidates(sensitivity, ru)

        assert filtered_sensitivity == sensitivity
        for ru_target, ru_matrix in ru.items():
            assert np.array_equal(filtered_ru[ru_target], ru_matrix)
        assert candidate_index_mapping == {'n1': [0, 1], 'n2': [0]}
        warning_mock.assert_not_called()

    @pytest.mark.parametrize('non_finite_values', [
        (np.nan, np.inf, -np.inf),
        (float('nan'), float('inf'), float('-inf')),
    ])
    def test_filter_non_finite_candidates(self, non_finite_values):
        """Test that non-finite candidates are removed from sensitivity and all RU targets."""
        nan_value, positive_inf, negative_inf = non_finite_values
        sensitivity = {
            'n1': [0.1, nan_value, 0.3],
            'n2': [positive_inf, 0.2, negative_inf],
        }
        ru = {
            RUTarget.WEIGHTS: np.array([[10], [11], [12], [13], [14], [15]]),
            RUTarget.ACTIVATION: np.array([[20, 21], [22, 23], [24, 25],
                                           [26, 27], [28, 29], [30, 31]]),
            RUTarget.TOTAL: np.array([[40, 41], [42, 43], [44, 45],
                                      [46, 47], [48, 49], [50, 51]]),
            RUTarget.BOPS: np.array([[60], [61], [62], [63], [64], [65]]),
        }

        filtered_sensitivity, filtered_ru, candidate_index_mapping = \
            MixedPrecisionIntegerLPSolver._filter_non_finite_candidates(sensitivity, ru)

        assert filtered_sensitivity == {'n1': [0.1, 0.3], 'n2': [0.2]}
        assert np.array_equal(filtered_ru[RUTarget.WEIGHTS], np.array([[10], [12], [14]]))
        assert np.array_equal(filtered_ru[RUTarget.ACTIVATION], np.array([[20, 21], [24, 25], [28, 29]]))
        assert np.array_equal(filtered_ru[RUTarget.TOTAL], np.array([[40, 41], [44, 45], [48, 49]]))
        assert np.array_equal(filtered_ru[RUTarget.BOPS], np.array([[60], [62], [64]]))
        assert candidate_index_mapping == {'n1': [0, 2], 'n2': [1]}

    def test_warn_on_non_finite_candidates(self, mocker):
        """Test that the warning identifies the layer, original indices, and non-finite values."""
        warning_mock = mocker.patch.object(Logger, 'warning')

        MixedPrecisionIntegerLPSolver._filter_non_finite_candidates(
            {'n1': [np.nan, 0.1, np.inf, -np.inf]},
            {RUTarget.WEIGHTS: np.array([[10], [11], [12], [13]])},
        )

        warning_mock.assert_called_once_with(
            'Ignoring mixed-precision candidates with non-finite sensitivity for layer n1: '
            'indices=[0, 2, 3], values=[nan, inf, -inf]')

    def test_raise_error_when_all_candidates_are_non_finite(self):
        """Test that a layer with no finite candidate raises an informative error."""
        with pytest.raises(ValueError, match='All mixed-precision candidates have non-finite sensitivity for layer n1'):
            MixedPrecisionIntegerLPSolver._filter_non_finite_candidates(
                {
                    'n1': [np.nan, np.inf, -np.inf],
                    'n2': [0.1, 0.2],
                },
                {RUTarget.WEIGHTS: np.array([[10], [11], [12], [13], [14]])},
            )

    @pytest.mark.parametrize('ru_matrix', [np.array([10, 11]), np.array([[10]])])
    def test_raise_error_for_invalid_ru_matrix(self, ru_matrix):
        """Test that invalid RU dimensions or candidate row counts raise an error."""
        with pytest.raises(ValueError, match='Resource utilization matrix'):
            MixedPrecisionIntegerLPSolver._filter_non_finite_candidates(
                {'n1': [0.1, 0.2]},
                {RUTarget.WEIGHTS: ru_matrix},
            )

    @pytest.mark.parametrize('ru_target', [RUTarget.WEIGHTS, RUTarget.BOPS])
    def test_run_with_non_finite_sensitivity(self, ru_target):
        """Test scalar RU targets return original indices after filtering non-finite sensitivities."""
        sensitivity = {
            'n1': [np.nan, 0.2, 0.1],
            'n2': [0.3, np.inf],
        }
        ru = {ru_target: np.array([[10], [2], [1], [3], [10]])}
        ru_constraints = {ru_target: np.array([4])}

        self._run_test(sensitivity, ru, ru_constraints, {'n1': 2, 'n2': 0})

    def test_run_with_non_finite_sensitivity_for_multi_column_ru(self):
        """Test that filtering preserves ACTIVATION and TOTAL constraints simultaneously."""
        sensitivity = {
            'n1': [np.nan, 0.2, 0.1],
            'n2': [0.3, np.inf],
        }
        ru = {
            RUTarget.ACTIVATION: np.array([[10, 10], [2, 2], [1, 1], [3, 3], [10, 10]]),
            RUTarget.TOTAL: np.array([[20, 20], [4, 4], [2, 2], [6, 6], [20, 20]]),
        }
        ru_constraints = {
            RUTarget.ACTIVATION: np.array([4, 4]),
            RUTarget.TOTAL: np.array([8, 8]),
        }

        self._run_test(sensitivity, ru, ru_constraints, {'n1': 2, 'n2': 0})

    @pytest.mark.skipif(not hasattr(pulp, 'LpSolveStatus'), reason='Requires PuLP 4.0 solve results')
    @pytest.mark.parametrize('status_name', ['TimeLimit', 'GapLimit'])
    @pytest.mark.parametrize('has_solution', [True, False], ids=['with_solution', 'without_solution'])
    def test_run_with_solver_limit(self, mocker, status_name, has_solution):
        """Accept feasible solutions at time or gap limits, but reject limits without a solution."""
        ilp_solver = MixedPrecisionIntegerLPSolver(
            {'n1': [np.nan, 0.2, 0.1]},
            {RUTarget.WEIGHTS: np.array([[10], [2], [1]])},
            {RUTarget.WEIGHTS: np.array([2])},
        )

        def solve_with_limit(lp_problem, solver):
            if has_solution:
                variables = lp_problem.variablesDict()
                variables['layer_0_0'].varValue = 0.0
                variables['layer_0_1'].varValue = 1.0
            return SimpleNamespace(status=getattr(pulp.LpSolveStatus, status_name), has_solution=has_solution)

        mocker.patch.object(pulp.LpProblem, 'solve', autospec=True, side_effect=solve_with_limit)

        if has_solution:
            assert ilp_solver.run() == {'n1': 2}
        else:
            with pytest.raises(RuntimeError, match='No solution was found for the LP problem'):
                ilp_solver.run()
            assert all(v.varValue is None for v in ilp_solver.layer_to_indicator_vars['n1'])

    def _run_test(self, sensitivity, ru, ru_constraints, exp_res):
        solver = MixedPrecisionIntegerLPSolver(sensitivity, ru, ru_constraints)
        res = solver.run()
        assert res == exp_res
