#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

import unittest
import numpy as np

from model_compression_toolkit.xquant.keras.similarity_functions import KerasSimilarityFunctions


class TestKerasSimilarityFunctions(unittest.TestCase):

    def test_compute_mse(self):
        # Test case 1
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)
        # Expected MSE = mean((1-1)^2 + (2-2)^2 + (3-3)^2) = 0
        expected_mse = 0.0
        self.assertAlmostEqual(KerasSimilarityFunctions.compute_mse(x, y), expected_mse)

        # Test case 2
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([4, 5, 6], dtype=np.float32)
        # Expected MSE = mean((1-4)^2 + (2-5)^2 + (3-6)^2) = mean(9 + 9 + 9) = 27/3 = 9
        expected_mse = 9.0
        self.assertAlmostEqual(KerasSimilarityFunctions.compute_mse(x, y), expected_mse)

    def test_compute_cs(self):
        # Test case 1
        x = np.array([1, 0], dtype=np.float32)
        y = np.array([1, 0], dtype=np.float32)
        # Expected Cosine Similarity = - (dot([1, 0], [1, 0]) / (||[1, 0]|| * ||[1, 0]||)) = -1
        expected_cs = -1.0
        self.assertAlmostEqual(KerasSimilarityFunctions.compute_cs(x, y), expected_cs)

        # Test case 2
        x = np.array([1, 0], dtype=np.float32)
        y = np.array([0, 1], dtype=np.float32)
        # Expected Cosine Similarity = - (dot([1, 0], [0, 1]) / (||[1, 0]|| * ||[0, 1]||)) = 0
        expected_cs = 0.0
        self.assertAlmostEqual(KerasSimilarityFunctions.compute_cs(x, y), expected_cs)

    def test_compute_sqnr(self):
        # Test case 1
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)
        # Expected SQNR = mean([1, 2, 3]^2) / mean([0, 0, 0]^2) = inf (since noise power is 0)
        expected_sqnr = float('inf')
        self.assertAlmostEqual(KerasSimilarityFunctions.compute_sqnr(x, y), expected_sqnr)

        # Test case 2
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([1, 1, 1], dtype=np.float32)
        # Expected signal power = mean([1^2, 2^2, 3^2]) = mean([1, 4, 9]) = 14/3
        # Expected noise power = mean([(1-1)^2, (2-1)^2, (3-1)^2]) = mean([0, 1, 4]) = 5/3
        # Expected SQNR = (14/3) / (5/3) = 14/5 = 2.8
        expected_sqnr = 14 / 5
        self.assertAlmostEqual(KerasSimilarityFunctions.compute_sqnr(x, y), expected_sqnr)


if __name__ == '__main__':
    unittest.main()

