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
from enum import Enum, auto
from typing import List, Callable, Optional

from model_compression_toolkit.constants import MP_DEFAULT_NUM_SAMPLES, ACT_HESSIAN_DEFAULT_BATCH_SIZE


class MpDistanceWeighting(Enum):
    """

    Defines interest points distances weighting methods.

    AVG - take the average distance over all interest points.

    LAST_LAYER - take only the distance of the last interest point.

    EXP - weighted average with weights based on exponent of negative distances between activations of the quantized and the float models.

    HESSIAN - weighted average with Hessians as weights.

    """
    AVG = auto()
    LAST_LAYER = auto()
    EXP = auto()
    HESSIAN = auto()


class MpMetricNormalization(Enum):
    """

    MAXBIT: normalize sensitivity metrics of layer candidates by max-bitwidth candidate (of that layer).

    MINBIT: normalize sensitivity metrics of layer candidates by min-bitwidth candidate (of that layer).

    NONE: no normalization.

    """
    MAXBIT = 'MAXBIT'
    MINBIT = 'MINBIT'
    NONE = 'NONE'


@dataclass
class MixedPrecisionQuantizationConfig:
    """
    Class with mixed precision parameters to quantize the input model.

    Args:
        compute_distance_fn (Callable): Function to compute a distance between two tensors. If None, using pre-defined distance methods based on the layer type for each layer.
        distance_weighting_method (MpDistanceWeighting): distance weighting method to use. By default, MpDistanceWeighting.AVG.
        num_of_images (int): Number of images to use to evaluate the sensitivity of a mixed-precision model comparing to the float model.
        configuration_overwrite (List[int]): A list of integers that enables overwrite of mixed precision with a predefined one.
        num_interest_points_factor (float): A multiplication factor between zero and one (represents percentage) to reduce the number of interest points used to calculate the distance metric.
        use_hessian_based_scores (bool): Whether to use Hessian-based scores for weighted average distance metric
          computation. This is identical to passing distance_weighting_method=MpDistanceWeighting.HESSIAN.
        norm_scores (bool): Whether to normalize the returned scores for the weighted distance metric (to get values between 0 and 1).
        refine_mp_solution (bool): Whether to try to improve the final mixed-precision configuration using a greedy algorithm that searches layers to increase their bit-width, or not.
        metric_normalization_threshold (float): A threshold for checking the mixed precision distance metric values, In case of values larger than this threshold, the metric will be scaled to prevent numerical issues.
        hessian_batch_size (int): The Hessian computation batch size. used only if using mixed precision with Hessian-based objective.
        metric_normalization (MpMetricNormalization): Metric normalization method.
        metric_epsilon (float | None): ensure minimal distance between the metric for any non-max-bidwidth candidate
          and a max-bitwidth candidate, i.e. metric(non-max-bitwidth) >= metric(max-bitwidth) + epsilon.
          If none, the computed metrics are used as is.
        exp_distance_weighting_sigma (float): sigma for exponential weighting method. A distance for each interest point
          is normalized by sigma prior to applying exponent.
        custom_metric_fn (Callable): Function to compute a custom metric. As input gets the model_mp and returns a
          float value for metric. If None, uses interest point metric.

    """
    compute_distance_fn: Optional[Callable] = None
    distance_weighting_method: MpDistanceWeighting = None
    num_of_images: int = MP_DEFAULT_NUM_SAMPLES
    configuration_overwrite: Optional[List[int]] = None
    num_interest_points_factor: float = field(default=1.0, metadata={"description": "Should be between 0.0 and 1.0"})
    use_hessian_based_scores: bool = False
    norm_scores: bool = True
    refine_mp_solution: bool = True
    metric_normalization_threshold: float = 1e10
    hessian_batch_size: int = ACT_HESSIAN_DEFAULT_BATCH_SIZE
    metric_normalization: MpMetricNormalization = MpMetricNormalization.NONE
    metric_epsilon: Optional[float] = 1e-6
    exp_distance_weighting_sigma: float = 0.1
    custom_metric_fn: Optional[Callable] = None
    _is_mixed_precision_enabled: bool = field(init=False, default=False)

    def __post_init__(self):
        # Validate num_interest_points_factor
        assert 0.0 < self.num_interest_points_factor <= 1.0, \
            "num_interest_points_factor should represent a percentage of " \
            "the base set of interest points that are required to be " \
            "used for mixed-precision metric evaluation, " \
            "thus, it should be between 0 to 1"
        if self.use_hessian_based_scores:
            assert self.distance_weighting_method in [None, MpDistanceWeighting.HESSIAN], \
                f'Distance method {self.distance_weighting_method} is incompatible with use_hessian_based_scores=True'
            self.distance_weighting_method = MpDistanceWeighting.HESSIAN
        elif self.distance_weighting_method is None and self.custom_metric_fn is None:
            self.distance_weighting_method = MpDistanceWeighting.AVG
        assert self.exp_distance_weighting_sigma > 0, (f'exp_distance_weighting_sigma should be positive, but got '
                                                       f'{self.exp_distance_weighting_sigma}')

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
