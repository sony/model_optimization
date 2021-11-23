# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import List, Callable

from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig, DEFAULTCONFIG
from model_compression_toolkit.common.similarity_analyzer import compute_mse


class MixedPrecisionMetricsWeighting(Enum):
    """
    Method for weighting distances among different layers when computing the total distance of a mixed-precision model from the float model:

    AVERAGE - All layers get the same weight, and the distance of a mixed-precision model from the float model is the average of distances among all layers (between the mixed-precision model and the float model).

    LAST_LAYER - The distance of a mixed-precision model from the float model is the distance between the last layer of the models.

    """
    AVERAGE = 0
    LAST_LAYER = 1


class MixedPrecisionQuantizationConfig(QuantizationConfig):

    def __init__(self,
                 qc: QuantizationConfig = DEFAULTCONFIG,
                 weights_n_bits: List[int] = None,
                 compute_distance_fn: Callable = compute_mse,
                 distance_weighting_method: MixedPrecisionMetricsWeighting = MixedPrecisionMetricsWeighting.AVERAGE):

        """
        Class to wrap all different parameters the library quantize the input model according to.
        Unlike QuantizationConfig, number of bits for quantization is a list of possible bit widths to
        support mixed-precision model quantization.

        Args:
            qc (QuantizationConfig): QuantizationConfig object containing parameters of how the model should be quantized.
            weights_n_bits (int): List of possible number of bits to quantize the coefficients.
            compute_distance_fn (Callable): Function to compute a distance between two tensors. We use it to estimate the distance of a mixed-precision model from the float model.
            distance_weighting_method (MixedPrecisionMetricsWeighting): Method to use when weighting the distances among different layers when computing the distance metric.
        """

        super().__init__(**qc.__dict__)
        self.weights_n_bits = weights_n_bits if weights_n_bits is not None else [qc.weights_n_bits]
        self.compute_distance_fn = compute_distance_fn
        self.distance_weighting_method = distance_weighting_method


# Default quantization configuration the library use.
DEFAULT_MIXEDPRECISION_CONFIG = MixedPrecisionQuantizationConfig(DEFAULTCONFIG,
                                                                 [8],
                                                                 compute_mse,
                                                                 MixedPrecisionMetricsWeighting.AVERAGE)
