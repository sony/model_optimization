# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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

from model_compression_toolkit.core.pytorch.back2framework.quantization_wrapper.wrapper_quantize_config import \
    WrapperQuantizeConfig


class NoQuantizationQuantizeConfig(WrapperQuantizeConfig):
    """
    QuantizationConfig for Fully Quantized model to keep layer's activations and weights un-quantized.
    """

    def __init__(self):
        super().__init__(is_weight_quantized=False,
                         is_activation_quantized=False)

    def get_weight_quantizers(self):
        """

        Returns: An empty list as this QC does not quantize the layer's weights.

        """
        return []

    def get_activation_quantizers(self):
        """

        Returns: An empty list as this QC does not quantize the layer's activations.

        """
        return []



