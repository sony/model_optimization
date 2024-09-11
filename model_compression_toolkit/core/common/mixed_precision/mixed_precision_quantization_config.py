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

from dataclasses import dataclass, field
from typing import List, Callable, Optional
from model_compression_toolkit.constants import MP_DEFAULT_NUM_SAMPLES, ACT_HESSIAN_DEFAULT_BATCH_SIZE
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import MpDistanceWeighting


@dataclass
class MixedPrecisionQuantizationConfig:
    """
    Class with mixed precision parameters to quantize the input model.

    Args:
        compute_distance_fn (Callable): Function to compute a distance between two tensors. If None, using pre-defined distance methods based on the layer type for each layer.
        distance_weighting_method (MpDistanceWeighting): MpDistanceWeighting enum value that provides a function to use when weighting the distances among different layers when computing the sensitivity metric.
        num_of_images (int): Number of images to use to evaluate the sensitivity of a mixed-precision model comparing to the float model.
        configuration_overwrite (List[int]): A list of integers that enables overwrite of mixed precision with a predefined one.
        num_interest_points_factor (float): A multiplication factor between zero and one (represents percentage) to reduce the number of interest points used to calculate the distance metric.
        use_hessian_based_scores (bool): Whether to use Hessian-based scores for weighted average distance metric computation.
        norm_scores (bool): Whether to normalize the returned scores for the weighted distance metric (to get values between 0 and 1).
        refine_mp_solution (bool): Whether to try to improve the final mixed-precision configuration using a greedy algorithm that searches layers to increase their bit-width, or not.
        metric_normalization_threshold (float): A threshold for checking the mixed precision distance metric values, In case of values larger than this threshold, the metric will be scaled to prevent numerical issues.
        hessian_batch_size (int): The Hessian computation batch size. used only if using mixed precision with Hessian-based objective.
    """

    compute_distance_fn: Optional[Callable] = None
    distance_weighting_method: MpDistanceWeighting = MpDistanceWeighting.AVG
    num_of_images: int = MP_DEFAULT_NUM_SAMPLES
    configuration_overwrite: Optional[List[int]] = None
    num_interest_points_factor: float = field(default=1.0, metadata={"description": "Should be between 0.0 and 1.0"})
    use_hessian_based_scores: bool = False
    norm_scores: bool = True
    refine_mp_solution: bool = True
    metric_normalization_threshold: float = 1e10
    hessian_batch_size: int = ACT_HESSIAN_DEFAULT_BATCH_SIZE
    _is_mixed_precision_enabled: bool = field(init=False, default=False)

    def __post_init__(self):
        # Validate num_interest_points_factor
        assert 0.0 < self.num_interest_points_factor <= 1.0, \
            "num_interest_points_factor should represent a percentage of " \
            "the base set of interest points that are required to be " \
            "used for mixed-precision metric evaluation, " \
            "thus, it should be between 0 to 1"

    def set_mixed_precision_enable(self):
        """
        Set a flag in mixed precision config indicating that mixed precision is enabled.
        """
        self._is_mixed_precision_enabled = True

    @property
    def is_mixed_precision_enabled(self):
        """
        A property that indicates whether mixed precision quantization is enabled.

        Returns: True if mixed precision quantization is enabled
        """
        return self._is_mixed_precision_enabled
