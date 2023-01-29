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
from typing import Any, List, Callable

from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.gptq.common.gptq_constants import REG_DEFAULT, REGULARIZATION_FUNCTION


class GPTQQuantizerConfig:
    """
    A base class to define specific quantizer configuration for GPTQ quantizer.
    """

    def __init__(self):
        self.n_batches = None

    def get_regularization_value(self, fxp_model: Any, **kwargs) -> float:
        """
        Computes a regularization value for the quantizer's loss (if needed).
        In the base class it only returns 0, to be used for GPTQ quantizers that doesn't require regularization.

        Args:
            fxp_model: The quantized model that is being trained.
            **kwargs: Additional arguments for the quantizer regularization computation.

        Returns: The regularization value.
        """

        return 0


class SoftQuantizerConfig(GPTQQuantizerConfig):
    def __init__(self, entropy_regularization: float = REG_DEFAULT):
        """
        Initialize object that holds the arguments that are needed for soft rounding quantizer.

        Args:
            entropy_regularization (float): A floating point number that defines the gumbel entropy regularization factor.
        """

        super().__init__()
        self.entropy_regularization = entropy_regularization


    def get_regularization_value(self, fxp_model: Any, **kwargs) -> float:
        """
        Computes a regularization value for the soft quantizer.

        Args:
            fxp_model: The quantized model that is being trained.
            **kwargs: Additional arguments for the quantizer regularization computation.

        Returns: The regularization value.
        """

        soft_rounding_reg_func = kwargs.get(REGULARIZATION_FUNCTION)

        if soft_rounding_reg_func is None:
            Logger.error("No regularization function has been given for computing the regularization "  # pragma: no cover
                         "of the soft quantizer.")
        if not isinstance(soft_rounding_reg_func, Callable):
            Logger.error("The provided regularization function of the soft quantizer is not compatible.")  # pragma: no cover

        soft_quant_list = soft_rounding_reg_func(fxp_model)
        reg = 0

        for sq in soft_quant_list:
            reg += sq

        return self.entropy_regularization * reg

    def set_num_batches(self, num_batches: int):
        """
        Allows to set the number of batches that the soft quantizer is using for training (in each epoch).

        Args:
            num_batches: number of batches to be set.

        """
        self.n_batches = num_batches
