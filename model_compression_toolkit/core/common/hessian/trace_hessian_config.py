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
    Configuration for Hessian-based information computation, specifically for the trace of the Hessian.

    This class is used to define parameters for the approximation of the trace of the Hessian matrix.
    It's important to note that this does not compute the actual Hessian but approximates some Hessian-based data.
    The computation can be computationally heavy and is based on the trace of the Hessian.

    """

    def __init__(self,
                 alpha: float = HESSIAN_OUTPUT_ALPHA,
                 num_iterations: int = HESSIAN_NUM_ITERATIONS
                 ):
        """
        Attributes:
            alpha (float): A tuning parameter to calibrate the contribution of the output feature maps.
            num_iterations (int): Number of iterations for computation approximation.
        """
        self.alpha = alpha
        self.num_iterations = num_iterations

    def __eq__(self, other):
        """
        Checks the equality of this configuration with another object.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the other object is an instance of TraceHessianConfig and has the same attributes, False otherwise.
        """
        if isinstance(other, TraceHessianConfig):
            return (self.alpha == other.alpha and
                    self.num_iterations == other.num_iterations
                    )
        return False

    def __hash__(self):
        """
        Computes a hash value for this configuration based on its attributes.

        Returns:
            The hash value of this configuration.
        """
        return hash((self.alpha,
                     self.num_iterations
                     ))
