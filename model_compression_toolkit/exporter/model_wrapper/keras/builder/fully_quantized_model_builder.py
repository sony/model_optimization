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

from typing import Tuple, Callable, Union
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Graph
from model_compression_toolkit.verify_packages import FOUND_TF
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.logger import Logger
import model_compression_toolkit.core as C

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
    from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
    from mct_quantizers import KerasQuantizationWrapper
    from mct_quantizers import KerasActivationQuantizationHolder
    from mct_quantizers.common.constants import OP_CALL_ARGS, OP_CALL_KWARGS

    def _get_wrapper(node: Union[common.BaseNode, FunctionalNode],
                     layer: Layer,
                     fw_impl=None) -> Layer:
        """
        A function which takes a computational graph node and a keras layer and perform the quantization wrapping
        Args:
            node: A node of mct graph.
            layer: A keras layer

        Returns: Wrapped layer with weights quantizers and activation quantizers

        """
        weights_quantizers, _ = fw_impl.get_inferable_quantizers(node)
        if len(weights_quantizers) > 0:
            # for positional weights we need to extract the weight's value.
            weights_values = {attr: node.get_weights_by_keys(attr)
                              for attr in weights_quantizers if isinstance(attr, int)}
            # When wrapping functional nodes, need to set call args\kwargs in wrapper, because they
            # are used during wrapper call method.
            func_node_kwargs = {OP_CALL_ARGS: node.op_call_args,
                                OP_CALL_KWARGS: node.op_call_kwargs
                                } if isinstance(node, FunctionalNode) else {}
            return KerasQuantizationWrapper(layer,
                                            weights_quantizers,
                                            weights_values,
                                            is_inputs_as_list=node.inputs_as_list,
                                            **func_node_kwargs)
        return layer


    def get_activation_quantizer_holder(node: common.BaseNode, fw_impl) -> Callable:
        """
        Retrieve a ActivationQuantizationHolder layer to use for activation quantization for a node.

        Args:
            node: Node to get ActivationQuantizationHolder to attach in its output.

        Returns:
            A ActivationQuantizationHolder layer for the node activation quantization.
        """
        _, activation_quantizers = fw_impl.get_inferable_quantizers(node)

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return KerasActivationQuantizationHolder(activation_quantizers[0])

        Logger.critical(
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
                                                        wrapper=lambda n, kn:
                                                        _get_wrapper(n, kn,
                                                                     fw_impl=C.keras.keras_implementation.KerasImplementation()),
                                                        get_activation_quantizer_holder_fn=lambda n:
                                                        get_activation_quantizer_holder(n,
                                                                                        fw_impl=C.keras.keras_implementation.KerasImplementation())).build_model()
        exportable_model.trainable = False

        Logger.info("\nPlease run your accuracy evaluation on the exported quantized model to verify it's accuracy.\n"
                    "Checkout the FAQ and Troubleshooting pages for resolving common issues and improving the quantized model accuracy:\n"
                    "FAQ: https://github.com/sony/model_optimization/tree/main/FAQ.md\n"
                    "Quantization Troubleshooting: https://github.com/sony/model_optimization/tree/main/quantization_troubleshooting.md")
        return exportable_model, user_info
else:
    def get_exportable_keras_model(*args, **kwargs):  # pragma: no cover
        Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                        "get_exportable_keras_model. The 'tensorflow' package is missing or is installed with a "
                        "version higher than 2.15.")  # pragma: no cover
