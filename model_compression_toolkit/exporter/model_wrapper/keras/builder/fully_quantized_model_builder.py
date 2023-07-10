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

from typing import Tuple, Callable
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.logger import Logger
from mct_quantizers import KerasActivationQuantizationHolder

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
    from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizers import get_quantization_quantizers
    from mct_quantizers import KerasQuantizationWrapper

    def _get_wrapper(node: common.BaseNode,
                     layer: Layer) -> Layer:
        """
        A function which takes a computational graph node and a keras layer and perform the quantization wrapping
        Args:
            node: A node of mct graph.
            layer: A keras layer

        Returns: Wrapped layer with weights quantizers and activation quantizers

        """
        weights_quantizers, _ = get_quantization_quantizers(node)
        if len(weights_quantizers) > 0:
            return KerasQuantizationWrapper(layer,
                                            weights_quantizers)
        return layer


    def get_activation_quantizer_holder(node: common.BaseNode) -> Callable:
        """
        Retrieve a ActivationQuantizationHolder layer to use for activation quantization for a node.

        Args:
            node: Node to get ActivationQuantizationHolder to attach in its output.

        Returns:
            A ActivationQuantizationHolder layer for the node activation quantization.
        """
        _, activation_quantizers = get_quantization_quantizers(node)

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return KerasActivationQuantizationHolder(activation_quantizers[0])

        Logger.error(
            f'ActivationQuantizationHolder supports a single quantizer but {len(activation_quantizers)} quantizers '
            f'were found for node {node}')



    def get_exportable_keras_model(graph: Graph) -> Tuple[tf.keras.models.Model, UserInformation]:
        """
        Convert graph to an exportable Keras model (model with all quantization parameters).
        An exportable model can then be exported using model_exporter, to retrieve the
        final exported model.

        Args:
            graph: Graph to convert to an exportable Keras model.

        Returns:
            Exportable Keras model and user information.
        """
        exportable_model, user_info = KerasModelBuilder(graph=graph,
                                                        wrapper=_get_wrapper,
                                                        get_activation_quantizer_holder_fn=get_activation_quantizer_holder).build_model()
        exportable_model.trainable = False
        return exportable_model, user_info
else:
    def get_exportable_keras_model(*args, **kwargs):  # pragma: no cover
        Logger.error('Installing tensorflow is mandatory '
                     'when using get_exportable_keras_model. '
                     'Could not find Tensorflow package.')