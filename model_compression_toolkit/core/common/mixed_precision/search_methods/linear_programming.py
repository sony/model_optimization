# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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
from collections import defaultdict

import numpy as np
from pulp import *
from typing import Dict, Tuple, Any, List

from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import RUTarget

# Limit ILP solver runtime in seconds
SOLVER_TIME_LIMIT = 60


class MixedPrecisionIntegerLPSolver:
    """ Integer Linear Programming solver for Mixed Precision.

        Args:
            layer_to_sensitivity_mapping: sensitivity per candidate per layer.
            candidates_ru: resource utilization per candidate.
            ru_constraints: resource utilization constraints corresponding to 'candidates_ru'.
    """
    def __init__(self,
                 layer_to_sensitivity_mapping: Dict[Any, List[float]],
                 candidates_ru: Dict[RUTarget, np.ndarray],
                 ru_constraints: Dict[RUTarget, np.ndarray]):

        self.layer_to_sensitivity_mapping = layer_to_sensitivity_mapping
        self.candidates_ru = candidates_ru
        self.ru_constraints = ru_constraints

        self.layer_to_indicator_vars, self.objective_vars = self._init_problem_vars(layer_to_sensitivity_mapping)

    def run(self) -> Dict[Any, int]:
        """
        Build and solve an ILP optimization problem.

        Returns:
            A dictionary from layer to the index of the selected bitwidth candidate.
        """
        # Add all equations and inequalities that define the problem.
        lp_problem = self._formalize_problem()

        # Use default PULP solver. Limit runtime in seconds
        solver = PULP_CBC_CMD(timeLimit=SOLVER_TIME_LIMIT)
        lp_problem.solve(solver=solver)  # Try to solve the problem.

        if lp_problem.status != LpStatusOptimal:
            raise RuntimeError(f'No solution was found for the LP problem, with status {lp_problem.status}')

        # Take the bitwidth index only if its corresponding indicator is one.
        mp_config = {
            layer: [v.varValue for v in vars].index(1.) for layer, vars in self.layer_to_indicator_vars.items()
        }
        return mp_config

    @staticmethod
    def _init_problem_vars(layer_to_metrics_mapping: Dict[Any, List[float]]) -> Tuple[Dict[Any, List[LpVariable]],
                                                                                      List[LpVariable]]:
        """
        Initialize the LP problem variables: Variable for each layer as to the index of the bitwidth it should use,
        and a variable for each indicator for whether we use the former variable or not.

        Args:
            layer_to_metrics_mapping: Mapping from each layer's index (in the model) to a dictionary that maps the
            bitwidth index to the observed sensitivity of the model.

        Returns:
            A tuple of two dictionaries: One from a layer to the variable for the bitwidth problem,
            and the second for indicators for each variable.
        """

        layer_to_indicator_vars = defaultdict(list)
        objective_vars = []

        for layer_idx, (layer, bitwidth_metrics) in enumerate(layer_to_metrics_mapping.items()):
            layer_to_indicator_vars[layer] = [
                LpVariable(f"layer_{layer_idx}_{qc_idx}", lowBound=0, upBound=1, cat=LpInteger)
                for qc_idx, _ in enumerate(bitwidth_metrics)
            ]

            objective_vars.append(LpVariable(f"s_{layer_idx}", 0))

        return layer_to_indicator_vars, objective_vars

    def _formalize_problem(self) -> LpProblem:
        """
        Formalize the LP problem by defining all inequalities that define the solution space.

        Returns:
            The formalized LP problem.
        """

        lp_problem = LpProblem()  # minimization problem by default
        lp_problem += lpSum(self.objective_vars)

        for layer_sensitivity, layer_indicator_vars, obj_var in zip(self.layer_to_sensitivity_mapping.values(),
                                                                    self.layer_to_indicator_vars.values(),
                                                                    self.objective_vars):
            # Use every bitwidth for every layer with its indicator.
            lp_problem += lpSum(list(np.multiply(layer_indicator_vars, layer_sensitivity))) == obj_var

            # Constraint of only one indicator==1
            lp_problem += lpSum(layer_indicator_vars) == 1

        # Bound the feasible solution space with the desired resource utilization values.
        self._add_ru_constraints(lp_problem=lp_problem)

        return lp_problem

    def _add_ru_constraints(self, lp_problem: LpProblem):
        """
        Adding targets constraints for the Lp problem for the given target resource utilization.
        The update to the Lp problem object is done inplace.

        Args:
            lp_problem: An Lp problem object to add constraint to.
        """
        indicator_vars = list(itertools.chain(*self.layer_to_indicator_vars.values()))

        for target, ru_matrix in self.candidates_ru.items():
            # We expect 2d matrix of shape (num candidates, m). For cumulative metrics (weights, bops) m=1 - overall
            # utilization. For max metrics (activation, total) m=num memory elements (max element depends on configuration)
            assert ru_matrix.ndim == 2
            if target in [RUTarget.WEIGHTS, RUTarget.BOPS]:
                assert ru_matrix.shape[1] == 1

            indicated_ru_matrix = ru_matrix.T * np.array(indicator_vars)
            # build lp sum term over all candidates
            ru_vec = indicated_ru_matrix.sum(axis=1)

            # For cumulative metrics a single constraint is added, for max metrics a separate constraint
            # is added for each memory element (each element < target => max element < target).
            assert len(ru_vec) == len(self.ru_constraints[target])
            for v, c in zip(ru_vec, self.ru_constraints[target]):
                lp_problem += v <= c
