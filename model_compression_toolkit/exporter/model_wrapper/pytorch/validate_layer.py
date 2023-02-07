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

from model_compression_toolkit.quantizers_infrastructure import PytorchQuantizationWrapper, \
    BasePyTorchInferableQuantizer


def is_pytorch_layer_exportable(layer: Any) -> bool:
    """
    Check whether a torch Module is a valid exportable module or not.

    Args:
        layer: PyTorch module to check if considered to be valid for exporting.

    Returns:
        Check whether a PyTorch layer is a valid exportable layer or not.
    """
    if isinstance(layer, PytorchQuantizationWrapper):
        quantizers = list(layer.weights_quantizers.values())
        quantizers.extend(layer.activation_quantizers)
        if all([isinstance(q, BasePyTorchInferableQuantizer) for q in quantizers]):
            return True
    return False
