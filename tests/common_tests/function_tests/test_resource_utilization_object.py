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
mixed_ru = ResourceUtilization(activation_memory=5, bops=10)


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
        self.assertEqual(default_ru.get_summary_str(restricted=False), f"Weights memory: {np.inf}, "
                                                                       f"Activation memory: {np.inf}, "
                                                                       f"Total memory: {np.inf}, "
                                                                       f"BOPS: {np.inf}")
        self.assertEqual(default_ru.get_summary_str(restricted=True), "")

        self.assertEqual(mixed_ru.get_summary_str(restricted=False), f"Weights memory: {np.inf}, "
                                                                     "Activation memory: 5, "
                                                                     f"Total memory: {np.inf}, "
                                                                     "BOPS: 10")
        self.assertEqual(mixed_ru.get_summary_str(restricted=True), "Activation memory: 5, BOPS: 10")

    def test_ru_hold_constraints(self):
        self.assertTrue(default_ru.is_satisfied_by(custom_ru))
        self.assertFalse(custom_ru.is_satisfied_by(default_ru))
