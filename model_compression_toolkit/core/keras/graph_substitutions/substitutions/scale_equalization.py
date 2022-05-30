# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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


from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dense, Conv2DTranspose, \
    Activation, ReLU, ZeroPadding2D
from tensorflow.nn import relu

from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.graph.graph_matchers import NodeOperationMatcher, WalkMatcher, \
    NodeFrameworkAttrMatcher
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.substitutions.scale_equalization import BaseScaleEqualization
from model_compression_toolkit.core.keras.constants import ACTIVATION, KERNEL, BIAS
from model_compression_toolkit.core.keras.constants import RELU

# Match linear layers.
op2d_node = NodeOperationMatcher(DepthwiseConv2D) | \
            NodeOperationMatcher(Conv2D) | \
            NodeOperationMatcher(Conv2DTranspose) | \
            NodeOperationMatcher(Dense)


homogeneous_activation_nodes = op2d_node & NodeFrameworkAttrMatcher(ACTIVATION, RELU)

activation_nodes = NodeOperationMatcher(ReLU) | \
                   NodeOperationMatcher(relu)

zeropad_node = NodeOperationMatcher(ZeroPadding2D)

MATCHER = WalkMatcher([homogeneous_activation_nodes,
                       op2d_node])

MATCHER_WITH_PAD = WalkMatcher([homogeneous_activation_nodes,
                                zeropad_node,
                                op2d_node])

MATCHER_MID = WalkMatcher([op2d_node,
                           activation_nodes,
                           op2d_node])

MATCHER_MID_WITH_PAD = WalkMatcher([op2d_node,
                                    activation_nodes,
                                    zeropad_node,
                                    op2d_node])


class ScaleEqualization(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to Keras
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ScaleEqualization object.
        Args:
            quant_config: Quantization configuration.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER,
                         kernel_str=KERNEL, bias_str=BIAS)


class ScaleEqualizationWithPad(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear--> ZeroPadding--> Linear
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ScaleEqualizationWithPad object.
        Args:
            quant_config: Quantization configuration.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER_WITH_PAD,
                         kernel_str=KERNEL, bias_str=BIAS)


class ScaleEqualizationMidActivation(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear--> RelU--> Linear

    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ScaleEqualizationMidActivation object.
        Args:
            quant_config: Quantization configuration.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER_MID,
                         kernel_str=KERNEL, bias_str=BIAS)


class ScaleEqualizationMidActivationWithPad(BaseScaleEqualization):
    """
    Substitution extends BaseScaleEqualization to the case of Linear--> RelU--> ZeroPadding--> Linear
    """

    def __init__(self,
                 quant_config: QuantizationConfig,
                 fw_info: FrameworkInfo):
        """
        Initialize a ScaleEqualizationMidActivationWithPad object.
        Args:
            quant_config: Quantization configuration.
            fw_info: Information needed for quantization about the specific framework (e.g., kernel channels indices,
            groups of layers by how they should be quantized, etc.)
        """

        super().__init__(quant_config=quant_config, fw_info=fw_info, matcher_instance=MATCHER_MID_WITH_PAD,
                         kernel_str=KERNEL, bias_str=BIAS)
