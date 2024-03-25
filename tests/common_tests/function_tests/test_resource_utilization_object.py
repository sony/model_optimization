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
from model_compression_toolkit.core import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    RUTarget

default_ru = ResourceUtilization()
custom_ru = ResourceUtilization(1, 2, 3, 4)


class TestResourceUtilizationObject(unittest.TestCase):

    def test_default(self):
        self.assertTrue(default_ru.weights_memory, np.inf)
        self.assertTrue(default_ru.activation_memory, np.inf)
        self.assertTrue(default_ru.total_memory, np.inf)
        self.assertTrue(default_ru.bops, np.inf)

        self.assertTrue(custom_ru.weights_memory, 1)
        self.assertTrue(custom_ru.activation_memory, 2)
        self.assertTrue(custom_ru.total_memory, 3)
        self.assertTrue(custom_ru.bops, 4)

    def test_representation(self):
        self.assertEqual(repr(default_ru), f"Weights_memory: {np.inf}, "
                                           f"Activation_memory: {np.inf}, "
                                           f"Total_memory: {np.inf}, "
                                           f"BOPS: {np.inf}")

        self.assertEqual(repr(custom_ru), f"Weights_memory: {1}, "
                                          f"Activation_memory: {2}, "
                                          f"Total_memory: {3}, "
                                          f"BOPS: {4}")

    def test_ru_hold_constraints(self):
        self.assertTrue(default_ru.holds_constraints(custom_ru))
        self.assertFalse(custom_ru.holds_constraints(default_ru))
        self.assertFalse(custom_ru.holds_constraints({RUTarget.WEIGHTS: 1,
                                                      RUTarget.ACTIVATION: 1,
                                                      RUTarget.TOTAL: 1,
                                                      RUTarget.BOPS: 1}))
