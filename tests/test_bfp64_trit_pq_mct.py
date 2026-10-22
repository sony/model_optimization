import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "model_compression_toolkit",
            "core",
            "common",
            "quantization",
            "quantizers",
        )
    ),
)

from bfp64_trit_pq_mct import (
    compute_candidate_ru_pq,
    solve_mixed_precision_pq,
    threshold_is_power_of_two_frexp,
)


class TestMctBfp64TritPqOptimization(unittest.TestCase):
    def test_threshold_is_power_of_two_frexp(self):
        self.assertTrue(threshold_is_power_of_two_frexp([0.125, 0.25, 1.0, 2.0, 64.0]))
        self.assertFalse(threshold_is_power_of_two_frexp([1.0, 3.0, 4.0]))
        self.assertFalse(threshold_is_power_of_two_frexp(-2.0))
        self.assertFalse(threshold_is_power_of_two_frexp(0.0))

    def test_candidate_ru_bops_and_sram_reduction(self):
        num_weights = 64 * 1024
        num_acts = 16 * 1024
        num_macs = 4_000_000

        ru_int8 = compute_candidate_ru_pq(
            num_weights, num_acts, num_macs, weight_mode="int8", act_mode="int8"
        )
        ru_trit_bfp = compute_candidate_ru_pq(
            num_weights, num_acts, num_macs, weight_mode="trit_b158", act_mode="bfp64_i8"
        )
        # Weight SRAM reduction >= 3.4x vs INT8 (and 12.8x vs FP32)
        self.assertGreaterEqual(ru_int8.weight_bytes / ru_trit_bfp.weight_bytes, 3.4)
        # Effective BOPs reduction > 5.0x vs INT8xINT8
        self.assertGreater(ru_int8.effective_bops / ru_trit_bfp.effective_bops, 5.0)

    def test_mixed_precision_pq_solver_with_non_finite_candidates(self):
        # Candidates: [trit_b158 (2000B), int4 (3600B), bfp64_i8 (6500B)]
        sensitivities = {
            "backbone_conv1": [float("inf"), 0.12, 0.01],  # non-finite trit candidate filtered!
            "backbone_conv2": [0.08, 0.03, 0.005],
            "head_fc": [0.45, 0.10, 0.01],
        }
        bytes_map = {
            "backbone_conv1": [2000, 3600, 6500],
            "backbone_conv2": [2000, 3600, 6500],
            "head_fc": [2000, 3600, 6500],
        }
        # Budget 12100: optimal allocation is conv1=2 (6500B, 0.01), conv2=0 (2000B, 0.08),
        # head_fc=1 (3600B, 0.10) -> 12100B total, sensitivity = 0.19
        alloc = solve_mixed_precision_pq(sensitivities, bytes_map, max_weight_bytes=12_100)
        self.assertEqual(alloc["backbone_conv1"], 2)
        self.assertEqual(alloc["backbone_conv2"], 0)
        self.assertEqual(alloc["head_fc"], 1)


if __name__ == "__main__":
    unittest.main()
