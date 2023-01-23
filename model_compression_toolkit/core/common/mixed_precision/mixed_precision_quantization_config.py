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

from enum import Enum
from typing import List, Callable, Tuple

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.mixed_precision.distance_weighting import get_average_weights
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig, DEFAULTCONFIG
from model_compression_toolkit.core.common.similarity_analyzer import compute_mse


class MixedPrecisionQuantizationConfigV2:

    def __init__(self,
                 compute_distance_fn: Callable = None,
                 distance_weighting_method: Callable = get_average_weights,
                 num_of_images: int = 32,
                 configuration_overwrite: List[int] = None,
                 num_interest_points_factor: float = 1.0,
                 use_grad_based_weights: bool = True,
                 output_grad_factor: float = 0.1,
                 norm_weights: bool = True,
                 refine_mp_solution: bool = True):
        """
        Class with mixed precision parameters to quantize the input model.
        Unlike QuantizationConfig, number of bits for quantization is a list of possible bit widths to
        support mixed-precision model quantization.

        Args:
            compute_distance_fn (Callable): Function to compute a distance between two tensors.
            distance_weighting_method (Callable): Function to use when weighting the distances among different layers when computing the sensitivity metric.
            num_of_images (int): Number of images to use to evaluate the sensitivity of a mixed-precision model comparing to the float model.
            configuration_overwrite (List[int]): A list of integers that enables overwrite of mixed precision with a predefined one.
            num_interest_points_factor (float): A multiplication factor between zero and one (represents percentage) to reduce the number of interest points used to calculate the distance metric.
            use_grad_based_weights (bool): Whether to use gradient-based weights for weighted average distance metric computation.
            output_grad_factor (float): A tuning parameter to be used for gradient-based weights.
            norm_weights (bool): Whether to normalize the returned weights (to get values between 0 and 1).
            refine_mp_solution (bool): Whether to try to improve the final mixed-precision configuration using a greedy algorithm that searches layers to increase their bit-width, or not.

        """

        self.compute_distance_fn = compute_distance_fn
        self.distance_weighting_method = distance_weighting_method
        self.num_of_images = num_of_images
        self.configuration_overwrite = configuration_overwrite
        self.refine_mp_solution = refine_mp_solution

        assert 0.0 < num_interest_points_factor <= 1.0, "num_interest_points_factor should represent a percentage of " \
                                                        "the base set of interest points that are required to be " \
                                                        "used for mixed-precision metric evaluation, " \
                                                        "thus, it should be between 0 to 1"
        self.num_interest_points_factor = num_interest_points_factor

        self.use_grad_based_weights = use_grad_based_weights
        self.output_grad_factor = output_grad_factor
        self.norm_weights = norm_weights

        if use_grad_based_weights is True:
            Logger.info(f"Using gradient-based weights for mixed-precision distance metric with tuning factor "
                        f"{output_grad_factor}")


class MixedPrecisionQuantizationConfig(QuantizationConfig):

    def __init__(self,
                 qc: QuantizationConfig = DEFAULTCONFIG,
                 compute_distance_fn: Callable = compute_mse,
                 distance_weighting_method: Callable = get_average_weights,
                 num_of_images: int = 32,
                 configuration_overwrite: List[int] = None,
                 num_interest_points_factor: float = 1.0):
        """
        Class to wrap all different parameters the library quantize the input model according to.
        Unlike QuantizationConfig, number of bits for quantization is a list of possible bit widths to
        support mixed-precision model quantization.

        Args:
            qc (QuantizationConfig): QuantizationConfig object containing parameters of how the model should be quantized.
            compute_distance_fn (Callable): Function to compute a distance between two tensors.
            distance_weighting_method (Callable): Function to use when weighting the distances among different layers when computing the sensitivity metric.
            num_of_images (int): Number of images to use to evaluate the sensitivity of a mixed-precision model comparing to the float model.
            configuration_overwrite (List[int]): A list of integers that enables overwrite of mixed precision with a predefined one.
            num_interest_points_factor: A multiplication factor between zero and one (represents percentage) to reduce the number of interest points used to calculate the distance metric.

        """

        super().__init__(**qc.__dict__)
        self.compute_distance_fn = compute_distance_fn
        self.distance_weighting_method = distance_weighting_method
        self.num_of_images = num_of_images
        self.configuration_overwrite = configuration_overwrite

        assert 0.0 < num_interest_points_factor <= 1.0, "num_interest_points_factor should represent a percentage of " \
                                                        "the base set of interest points that are required to be " \
                                                        "used for mixed-precision metric evaluation, " \
                                                        "thus, it should be between 0 to 1"
        self.num_interest_points_factor = num_interest_points_factor

    def separate_configs(self) -> Tuple[QuantizationConfig, MixedPrecisionQuantizationConfigV2]:
        """
        A function to separate the old MixedPrecisionQuantizationConfig into QuantizationConfig
        and MixedPrecisionQuantizationConfigV2

        Returns: QuantizationConfig, MixedPrecisionQuantizationConfigV2

        """
        _dummy_quant_config = QuantizationConfig()
        _dummy_mp_config_experimental = MixedPrecisionQuantizationConfigV2()
        qc_dict = {}
        mp_dict = {}
        for k, v in self.__dict__.items():
            if hasattr(_dummy_quant_config, k):
                qc_dict.update({k: v})
            elif hasattr(_dummy_mp_config_experimental, k):
                mp_dict.update({k: v})
            else:
                Logger.error(f'Attribute "{k}" mismatch: exists in MixedPrecisionQuantizationConfig but not in '
                             f'MixedPrecisionQuantizationConfigV2')  # pragma: no cover

        return QuantizationConfig(**qc_dict), MixedPrecisionQuantizationConfigV2(**mp_dict)


# Default quantization configuration the library use.
DEFAULT_MIXEDPRECISION_CONFIG = MixedPrecisionQuantizationConfig(DEFAULTCONFIG,
                                                                 compute_mse,
                                                                 get_average_weights)
