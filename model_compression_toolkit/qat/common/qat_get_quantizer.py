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
from typing import Any

from model_compression_toolkit import TrainingMethod
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.quantizers_infrastructure import QuantizationTarget
from model_compression_toolkit.quantizers_infrastructure.common.constants import QUANTIZATION_TARGET, \
    QUANTIZATION_METHOD, QUANTIZER_TYPE
from model_compression_toolkit.quantizers_infrastructure.common.get_all_subclasses import get_all_subclasses


def get_quantizer_class(quant_target: QuantizationTarget,
                        quantizer_type: TrainingMethod,
                        quant_method: QuantizationMethod,
                        quantizer_base_class: type) -> type:
    """
    Searches for a quantizer class that matches the requested QuantizationTarget and QuantizationMethod and TrainingMethod.
    Exactly one class should be found.

    Args:
        quant_target: QuantizationTarget value which indicates what is the target for quantization to
            use the quantizer for.
        quantizer_type: The type of the quantizer (quantization technique).
            This can differ, depending on the purpose the quantizer is for.
        quant_method: A list of QuantizationMethod values to indicate all type of quantization methods that the
            quantizer supports.
        quantizer_base_class: A type of quantizer that the requested quantizer should inherit from.

    Returns: A class of a quantizer that inherits from BaseKerasQATTrainableQuantizer.

    """
    qat_quantizer_classes = get_all_subclasses(quantizer_base_class)
    filtered_quantizers = list(filter(lambda q_class: getattr(q_class, QUANTIZATION_TARGET) == quant_target and
                                                      getattr(q_class, QUANTIZATION_METHOD) is not None and
                                                       quant_method in getattr(q_class, QUANTIZATION_METHOD) and
                                                      getattr(q_class, QUANTIZER_TYPE) == quantizer_type,
                                      qat_quantizer_classes))

    if len(filtered_quantizers) != 1:
        Logger.error(f"Found {len(filtered_quantizers)} quantizer for target {quant_target.value} "
                     f"that matches the requested quantization method {quant_method.name} and "
                     f"quantizer type {quantizer_type.value} but there should be exactly one."
                     f"The possible quantizers that were found are {filtered_quantizers}.")

    return filtered_quantizers[0]
