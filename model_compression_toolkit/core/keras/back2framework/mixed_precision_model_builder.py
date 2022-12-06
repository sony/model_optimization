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
from typing import List, Tuple, Any, Dict

import tensorflow as tf
import tensorflow_model_optimization.quantization.keras.graph_transformations.model_transformer as mt
from keras.models import Model

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder, \
    is_layer_fake_quant, get_node_name_from_layer
from model_compression_toolkit.core.keras.quantizer.input_layer_quantize_transform import \
    InputLayerWrapperTransform

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_quantize_config import \
    SelectiveQuantizeConfig
from packaging import version

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.python.keras.layers.core import TFOpLambda, SlicingOpLambda  # pragma: no cover
else:
    from keras.layers.core import TFOpLambda, SlicingOpLambda

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.core.common.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.keras.quantizer.mixed_precision.quantization_config_factory import \
    quantization_config_builder_mixed_precision
from tensorflow.python.util.object_identity import Reference as TFReference


class MixedPrecisionKerasModelBuilder(KerasModelBuilder):
    """
    Builder of mixed-precision Keras models.
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
        if node.is_all_activation_candidates_equal():
            # otherwise, we want to use the float tensor when building the model for MP search
            return node.candidates_quantization_cfg[0].activation_quantization_cfg.quantize_node_output(input_tensors)
        return input_tensors

    def build_model(self) -> Tuple[Model, UserInformation]:
        """
        Build a Keras mixed-precision model and return it.
        Returns: Mixed-precision Keras model.

        """
        model, user_info = super().build_model()

        conf_nodes_names = self.graph.get_configurable_sorted_nodes_names()

        def _quantize_multiple_nbits(layer):
            node = self.oh.layer_to_node_dict.get(layer)
            if node is not None:
                # Wrap only if its weights should be quantized
                if node.name in conf_nodes_names:
                    # TODO: Maybe FullyQuantizedQuantizeWrapper to allow using TFOpLambda in MP
                    if node.layer_class in [TFOpLambda, SlicingOpLambda]:
                        Logger.critical(f"Activation mixed-precision is not supported for layers of type "  # pragma: no cover
                                        f"{node.layer_class}. Please modify the TargetPlatformModel object, "
                                        f"such that layers of type {node.layer_class} "
                                        f"won't have more than one quantization configuration option.")
                    return QuantizeWrapper(layer, quantization_config_builder_mixed_precision(node))
                return layer

            elif is_layer_fake_quant(layer):
                return layer
            else:
                raise Exception(  # pragma: no cover
                    f'Mismatch between keras model and graph cant find node named: '
                    f'{get_node_name_from_layer(layer)}')

        # clone each layer in the model and apply _quantize to the layer.
        model = tf.keras.models.clone_model(model,
                                            input_tensors=None,
                                            clone_function=_quantize_multiple_nbits)

        # We use a model transformer to wrap the input layer with QuantizeWrapper,
        # to allow layer configuration to different bitwidths.
        # A model transformer allows to modify a layer in an existing model, by applying the given list of
        # transformers on the model (in this case,
        # we only apply single transformer - InputLayerQuantizeTransform)
        model_inputs = self.graph.get_inputs()

        input_transformer = mt.ModelTransformer(model, [InputLayerWrapperTransform(inp,
                                                                                   quantization_config_builder_mixed_precision(inp),
                                                                                   self.get_custom_objects(),
                                                                                   QuantizeWrapper)
                                                        for inp in model_inputs])
        model = input_transformer.transform()[0]

        return model, user_info

    @staticmethod
    def get_custom_objects() -> Dict[str, Any]:
        """

        Returns: Dictionary of custom objects needed to load this model builder's output.

        """
        return {QuantizeWrapper.__name__:QuantizeWrapper,
                SelectiveQuantizeConfig.__name__: SelectiveQuantizeConfig}

