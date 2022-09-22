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
import copy
from typing import List, Tuple

import tensorflow as tf
from keras.models import Model
from tensorflow.python.util.object_identity import Reference as TFReference
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit import get_target_platform_capabilities
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.framework_implementation import FrameworkImplementation
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder, \
    is_layer_fake_quant, get_node_name_from_layer
from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
from model_compression_toolkit.qat.keras.quantizer.config_factory import quantization_config_builder

DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


def _is_qat_applicable(node: common.BaseNode,
                       fw_info: FrameworkInfo) -> bool:
    """
    A function for deciding if a layer should be fine-tuned during QAT
    Args:
        node (BaseNode): Node for quantization decision
        fw_info (FrameworkInfo): Keras quantization information

    Returns:
        A boolean whether the layer is to be wrapped with a QuantizeWrapper
    """

    return fw_info.is_kernel_op(node.type) and node.is_weights_quantization_enabled()


class QATKerasModelBuilder(KerasModelBuilder):
    """
    Builder of QAT Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """
        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[TFReference]) -> List[TFReference]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """
        return node.final_activation_quantization_cfg.quantize_node_output(input_tensors)

    def build_model(self) -> Tuple[Model, UserInformation]:
        """
        Build a Keras QAT model and return it.
        Returns: QAT Keras model.

        """
        #################################################
        # Prepare model for Quantization Aware Training
        #################################################

        # Quantize graph weights that are not to be fine-tuned during QAT
        model, user_info = super().build_model()

        # Wrap layers to be fine-tuned during QAT with QuantizeWrapper
        def _quantize(layer):
            node = self.oh.layer_to_node_dict.get(layer)
            if node is not None:
                if _is_qat_applicable(node, self.fw_info):
                    return QuantizeWrapper(layer,
                                           quantization_config_builder(node,
                                                                       self.fw_info))
                else:
                    return layer
            elif is_layer_fake_quant(layer):
                # A fake quant layer was added in the core_model_builder to quantize the activations
                return layer
            else:
                Logger.error(
                    f"Mismatch between keras model and graph can't find node named: {get_node_name_from_layer(layer)}")

        # clone each layer in the model and apply _quantize to the layer.
        qat_model = tf.keras.models.clone_model(model,
                                                input_tensors=None,
                                                clone_function=_quantize)

        return qat_model, user_info
