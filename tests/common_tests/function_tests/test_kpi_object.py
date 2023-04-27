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


import unittest
import numpy as np
from model_compression_toolkit.core import KPI
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPITarget

default_kpi = KPI()
custom_kpi = KPI(1, 2, 3, 4)


class TestKPIObject(unittest.TestCase):

    def test_default(self):
        self.assertTrue(default_kpi.weights_memory, np.inf)
        self.assertTrue(default_kpi.activation_memory, np.inf)
        self.assertTrue(default_kpi.total_memory, np.inf)
        self.assertTrue(default_kpi.bops, np.inf)

        self.assertTrue(custom_kpi.weights_memory, 1)
        self.assertTrue(custom_kpi.activation_memory, 2)
        self.assertTrue(custom_kpi.total_memory, 3)
        self.assertTrue(custom_kpi.bops, 4)

    def test_representation(self):
        self.assertEqual(repr(default_kpi), f"Weights_memory: {np.inf}, "
                                            f"Activation_memory: {np.inf}, "
                                            f"Total_memory: {np.inf}, "
                                            f"BOPS: {np.inf}")

        self.assertEqual(repr(custom_kpi), f"Weights_memory: {1}, "
                                           f"Activation_memory: {2}, "
                                           f"Total_memory: {3}, "
                                           f"BOPS: {4}")

    def test_kpi_hold_constraints(self):
        self.assertTrue(default_kpi.holds_constraints(custom_kpi))
        self.assertFalse(custom_kpi.holds_constraints(default_kpi))
        self.assertFalse(custom_kpi.holds_constraints({KPITarget.WEIGHTS: 1,
                                                       KPITarget.ACTIVATION: 1,
                                                       KPITarget.TOTAL: 1,
                                                       KPITarget.BOPS: 1}))
