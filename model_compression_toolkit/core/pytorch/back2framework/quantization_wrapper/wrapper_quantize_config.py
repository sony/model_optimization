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

class WrapperQuantizeConfig:
    """
    Configuration for how a layer should be quantized.
    """

    def __init__(self,
                 is_weight_quantized: bool,
                 is_activation_quantized: bool
                 ):
        """

        Args:
            is_weight_quantized: Whether weights quantization is enabled or not.
            is_activation_quantized: Whether activation quantization is enabled or not.
        """

        self.is_weight_quantized = is_weight_quantized
        self.is_activation_quantized = is_activation_quantized

    def get_weight_quantizers(self) -> list:
        """

        Returns: A List of quantizers for weights quantization.

        """
        raise NotImplemented


    def get_activation_quantizers(self) -> list:
        """

        Returns: A List of quantizers for activation quantization.

        """
        raise NotImplemented



