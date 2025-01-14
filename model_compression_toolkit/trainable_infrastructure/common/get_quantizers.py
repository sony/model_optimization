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
from typing import Union, Any

from model_compression_toolkit.logger import Logger
from mct_quantizers import QuantizationTarget, QuantizationMethod
from mct_quantizers.common.constants \
    import QUANTIZATION_TARGET, QUANTIZATION_METHOD, QUANTIZER_ID
from mct_quantizers.common.get_all_subclasses \
    import get_all_subclasses


def get_trainable_quantizer_class(quant_target: QuantizationTarget,
                                  quantizer_id: Any,
                                  quant_method: QuantizationMethod,
                                  quantizer_base_class: type) -> type:
    """
    Searches for a trainable quantizer class that matches the requested QuantizationTarget and QuantizationMethod and
    a task dedicated quantizer type. Exactly one class should be found.

    Args:
        quant_target: QuantizationTarget value which indicates what is the target for quantization to
            use the quantizer for.
        quantizer_id: A unique identifier for the quantizer class.
        quant_method: A list of QuantizationMethod values to indicate all type of quantization methods that the
            quantizer supports.
        quantizer_base_class: A type of quantizer that the requested quantizer should inherit from.

    Returns: A class of a quantizer that inherits from BaseKerasQATTrainableQuantizer.

    """
    qat_quantizer_classes = get_all_subclasses(quantizer_base_class)
    if len(qat_quantizer_classes) == 0:
        Logger.critical(f"No quantizer classes inherited from {quantizer_base_class} were detected.")  # pragma: no cover

    filtered_quantizers = list(filter(lambda q_class: getattr(q_class, QUANTIZATION_TARGET, None) is not None and
                                                      getattr(q_class, QUANTIZATION_TARGET) == quant_target and
                                                      getattr(q_class, QUANTIZATION_METHOD, None) is not None and
                                                       quant_method in getattr(q_class, QUANTIZATION_METHOD, []) and
                                                      getattr(q_class, QUANTIZER_ID, None) == quantizer_id,
                                      qat_quantizer_classes))

    if len(filtered_quantizers) != 1:
        Logger.critical(f"Found {len(filtered_quantizers)} quantizers for target {quant_target.value}, "
                        f"matching the requested quantization method {quant_method.name} and "
                        f"quantizer type {quantizer_id.value}, but exactly one is required. "
                        f"Identified quantizers: {filtered_quantizers}.")

    return filtered_quantizers[0]
