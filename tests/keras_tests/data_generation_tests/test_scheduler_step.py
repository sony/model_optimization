# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from keras.optimizers import Adam

from model_compression_toolkit.data_generation.keras.optimization_functions.lr_scheduler import ReduceLROnPlateau


class TestReduceLROnPlateau(unittest.TestCase):
    def setUp(self):
        self.optimizer = Adam(learning_rate=0.1)
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer)

    def test_initialization(self):
        self.assertEqual(self.scheduler.factor, 0.1)
        self.assertEqual(self.scheduler.patience, 10)
        self.assertEqual(self.scheduler.threshold, 1e-4)
        self.assertEqual(self.scheduler.cooldown, 0)
        self.assertEqual(self.scheduler.min_lr, 0)
        self.assertEqual(self.scheduler.cooldown_counter, 0)
        self.assertEqual(self.scheduler.best, np.Inf)

    def test_reset(self):
        self.scheduler._reset()
        self.assertEqual(self.scheduler.best, np.Inf)
        self.assertEqual(self.scheduler.cooldown_counter, 0)

    def test_in_cooldown(self):
        self.scheduler.cooldown_counter = 1
        self.assertTrue(self.scheduler.in_cooldown)
        self.scheduler.cooldown_counter = 0
        self.assertFalse(self.scheduler.in_cooldown)

    def test_learning_rate_reduction(self):
        self.optimizer.learning_rate = 0.1
        self.scheduler._reset()
        for p in range(self.scheduler.patience + 2):
            self.scheduler.on_epoch_end(p, 0.1)
        self.assertLess(float(self.optimizer.learning_rate.numpy()), 0.1)
        self.assertEqual(self.scheduler.cooldown_counter, self.scheduler.cooldown)

    def test_minimum_learning_rate(self):
        self.optimizer.learning_rate = 0.1
        self.scheduler._reset()
        for p in range(100):
            self.scheduler.on_epoch_end(p, 0.1)
        self.assertGreaterEqual(float(self.optimizer.learning_rate.numpy()), self.scheduler.min_lr)

    def test_immediate_improvement(self):
        initial_lr = 0.1
        self.optimizer.learning_rate = initial_lr
        self.scheduler._reset()
        self.scheduler.on_epoch_end(1, 0.05)  # Immediate improvement
        self.assertAlmostEqual(float(self.optimizer.learning_rate.numpy()), initial_lr, places=7)


if __name__ == '__main__':
    unittest.main()
