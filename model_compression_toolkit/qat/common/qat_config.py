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

from typing import Dict
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.trainable_infrastructure import TrainingMethod


def is_qat_applicable(node: common.BaseNode) -> bool:
    """
    A function for deciding if a layer should be fine-tuned during QAT

    Args:
        node (BaseNode): Node for quantization decision

    Returns:
        A boolean whether the layer is to be wrapped with a QuantizeWrapper
    """
    return (node.kernel_attr is not None and node.is_weights_quantization_enabled(node.kernel_attr)) \
            or node.is_activation_quantization_enabled()


class QATConfig:
    """
    QAT configuration class.
    """

    def __init__(self, weight_training_method: TrainingMethod = TrainingMethod.STE,
                 activation_training_method: TrainingMethod = TrainingMethod.STE,
                 weight_quantizer_params_override: Dict = None,
                 activation_quantizer_params_override: Dict = None,
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
