# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
import model_compression_toolkit as mct
from model_compression_toolkit.core.common.pruning.pruning_info import unroll_simd_scores_to_per_channel_scores


class TestPruningInfo(unittest.TestCase):

    def setUp(self):
        # Setup some mock pruning masks and importance scores
        self.mock_pruning_masks = {"Layer1": np.array([1, 0, 1]),
                                   "Layer2": np.array([0, 1])}
        self.mock_importance_scores = {"Layer1": np.array([0.5, 0.3, 0.7]),
                                       "Layer2": np.array([0.2, 0.8])}
        self.pruning_info = mct.pruning.PruningInfo(self.mock_pruning_masks,
                                                    self.mock_importance_scores)

    def test_get_pruning_mask(self):
        # Test to check if the correct pruning masks are returned
        self.assertEqual(self.pruning_info.pruning_masks, self.mock_pruning_masks)

    def test_get_importance_score(self):
        # Test to check if the correct importance scores are returned
        self.assertEqual(self.pruning_info.importance_scores, self.mock_importance_scores)


class TestUnrollSIMDScores(unittest.TestCase):

    def test_unroll_simd_scores(self):
        # Setup mock SIMD scores and group indices
        simd_scores = {"Layer1": np.array([0.2, 0.4, 0.6])}
        simd_groups_indices = {"Layer1": [np.array([4, 1]), np.array([2, 3]), np.array([0])]}

        # Expected output
        expected_scores = {"Layer1": np.array([0.6, 0.2, 0.4, 0.4, 0.2])}

        # Test the unroll_simd_scores_to_per_channel_scores function
        result = unroll_simd_scores_to_per_channel_scores(simd_scores, simd_groups_indices)
        self.assertTrue(np.array_equal(result["Layer1"], expected_scores["Layer1"]))


if __name__ == '__main__':
    unittest.main()
