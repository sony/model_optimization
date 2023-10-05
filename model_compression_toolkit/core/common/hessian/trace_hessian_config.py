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
from model_compression_toolkit.constants import HESSIAN_OUTPUT_ALPHA, HESSIAN_NUM_ITERATIONS


class TraceHessianConfig:
    """
    Configuration class for Hessian computation.
    """

    def __init__(self,
                 alpha: float = HESSIAN_OUTPUT_ALPHA,
                 num_iterations: int = HESSIAN_NUM_ITERATIONS
                 ):
        """

        Args:
            alpha: A tuning parameter to allow calibration between the contribution of the output feature maps
            returned weights and the other feature maps weights (since the gradient of the output layers does not
            provide a compatible weight for the distance metric computation).
            num_iterations: Number of iterations for computation approximation.
        """
        self.alpha = alpha
        self.num_iterations = num_iterations

    def __eq__(self, other):
        """
        Checks equality of two HessianConfig objects.
        """
        if isinstance(other, TraceHessianConfig):
            return (self.alpha == other.alpha and
                    self.num_iterations == other.num_iterations
                    )
        return False

    def __hash__(self):
        """
        Computes the hash of the HessianConfig object for dictionary usage or other hashing requirements.
        """
        return hash((self.alpha,
                     self.num_iterations
                     ))
