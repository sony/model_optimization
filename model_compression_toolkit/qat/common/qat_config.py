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

from enum import Enum


class TrainingMethod(Enum):
    STE = 0


class QATConfig:
    """
    QAT configuration class.
    """

    def __init__(self, weight_training_method=TrainingMethod.STE,
                 activation_training_method=TrainingMethod.STE,
                 weight_quantizer_params_override=None,
                 activation_quantizer_params_override=None,
                 ):
        """

        Args:
            weight_training_method (TrainingMethod): Training method for weight quantizers
            activation_training_method (TrainingMethod): Training method for activation quantizers:
            weight_quantizer_params_override: A dictionary of parameters to override in weight quantization quantizer instantiation. Defaults to None (no parameters)
            activation_quantizer_params_override: A dictionary of parameters to override in activation quantization quantizer instantiation. Defaults to None (no parameters)
        """
        self.weight_training_method = weight_training_method
        self.activation_training_method = activation_training_method
        self.weight_quantizer_params_override = {} if weight_quantizer_params_override is None else weight_quantizer_params_override
        self.activation_quantizer_params_override = {} if activation_quantizer_params_override is None else activation_quantizer_params_override
