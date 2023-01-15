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
from model_compression_toolkit.core.common.constants import FOUND_TORCH
from model_compression_toolkit.quantizers_infrastructure import BaseInferableQuantizer, QuantizationTarget

if FOUND_TORCH:
    class BasePyTorchInferableQuantizer(BaseInferableQuantizer):
        def __init__(self,
                     quantization_target: QuantizationTarget):
            """
            This class is a base quantizer for PyTorch quantizers for inference only.

            Args:
                quantization_target: An enum which selects the quantizer tensor type: activation or weights.
            """
            super(BasePyTorchInferableQuantizer, self).__init__(quantization_target=quantization_target)
else:
    class BasePyTorchInferableQuantizer:
        def __init__(self, *args, **kwargs):
            raise Exception('Installing torch is mandatory '
                            'when using BasePyTorchInferableQuantizer. '
                            'Could not find torch package.')
