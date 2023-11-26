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
from typing import Tuple, Any, Dict, Union, List

from packaging import version
import tensorflow as tf
if version.parse(tf.__version__) >= version.parse("2.13"):
    from keras.src.engine.base_layer import Layer
else:
    from keras.engine.base_layer import Layer

from keras.models import Model
from mct_quantizers import KerasQuantizationWrapper, KerasActivationQuantizationHolder, QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder
from model_compression_toolkit.core.keras.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.keras.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer

from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizer import \
    get_inferable_quantizer_kwargs

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO


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

        self.graph = graph

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs,
                         wrapper=self.mixed_precision_wrapper,
                         get_activation_quantizer_holder_fn=self.mixed_precision_activation_holder)

    def mixed_precision_wrapper(self,
                                n: common.BaseNode,
                                layer: Layer) -> Union[KerasQuantizationWrapper, Layer]:
        """
        A function which takes a computational graph node and a keras layer and perform the quantization
        wrapping for mixed precision.

        Args:
            n: A node of mct graph.
            layer: A keras layer

        Returns: Wrapped layer with a configurable quantizer if the layer should quantized in mixed precision,
        otherwise returns either the layer wrapped with a fixed precision inferable quantizer or the layer as is if it's
        not supposed to be quantized.

        """

        weights_conf_nodes_names = [n.name for n in self.graph.get_weights_configurable_nodes()]

        if n.is_weights_quantization_enabled():
            kernel_attributes = self.fw_info.get_kernel_op_attributes(n.type)
            if n.name in weights_conf_nodes_names:
                return KerasQuantizationWrapper(layer,
                                                weights_quantizers={attr: ConfigurableWeightsQuantizer(
                                                    **self._get_weights_configurable_quantizer_kwargs(n, attr))
                                                    for attr in kernel_attributes})
            else:
                node_weights_qc = n.get_unique_weights_candidates()
                if not len(node_weights_qc) == 1:
                    Logger.error(f"Expecting node {n.name} to have a unique weights configuration "  # pragma: no cover
                                 f"but {len(node_weights_qc)} different configurations exist.")

                quantier_for_node = get_inferable_quantizer_class(QuantizationTarget.Weights,
                                                                  node_weights_qc[0].weights_quantization_cfg.weights_quantization_method,
                                                                  BaseKerasInferableQuantizer)
                kwargs = get_inferable_quantizer_kwargs(node_weights_qc[0].weights_quantization_cfg,
                                                        QuantizationTarget.Weights)

                return KerasQuantizationWrapper(layer,
                                                weights_quantizers={attr: quantier_for_node(**kwargs)
                                                                    for attr in kernel_attributes})

        return layer

    def _get_weights_configurable_quantizer_kwargs(self, n: BaseNode, attr: str) -> Dict[str, Any]:
        """
        Get the quantization parameters for a configurable quantizer.

        Args:
            n: The node for which the quantizer is being created.
            attr: The name of the weights attribute to be quantized.

        Returns:
            The quantization parameters as a dictionary.
        """

        assert n.candidates_quantization_cfg is not None, f"Node {n.name} candidates_quantization_cfg is None"
        node_q_cfg_candidates = n.candidates_quantization_cfg
        # sort by descending bit width so using indices would be easier
        node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.weights_n_bits,
                                                  x.activation_quantization_cfg.activation_n_bits), reverse=True)

        float_weights = n.get_weights_by_keys(attr)

        max_cfg_candidates = n.find_max_candidates_indices()
        if not len(max_cfg_candidates) == 1:
            Logger.error(f"A maximal config candidate must be defined, "  # pragma: no cover
                         f"but some node have multiple potential maximal candidates")

        max_candidate_idx = max_cfg_candidates[0]

        return {'node_q_cfg': node_q_cfg_candidates,
                'float_weights': float_weights,
                'max_candidate_idx': max_candidate_idx
                }

    def mixed_precision_activation_holder(self, n: BaseNode) -> KerasActivationQuantizationHolder:
        """
        Retrieve a KerasActivationQuantizationHolder layer to use for activation quantization for a node.
        The layer should hold either a configurable activation quantizer, if it is quantized with mixed precision,
        or an inferable quantizer for fixed single bit-width quantization.

        Args:
            n: Node to get KerasActivationQuantizationHolder to attach in its output.

        Returns:
            A KerasActivationQuantizationHolder layer for the node activation quantization.
        """

        activation_conf_nodes_names = [n.name for n in self.graph.get_activation_configurable_nodes()]

        activation_quantizers = []
        if n.is_activation_quantization_enabled():
            num_of_outputs = len(n.output_shape) if isinstance(n.output_shape, list) else 1
            if n.name in activation_conf_nodes_names:
                assert n.candidates_quantization_cfg is not None, f"Node {n.name} candidates_quantization_cfg is None"
                node_q_cfg_candidates = n.candidates_quantization_cfg
                # sort by descending bit width so using indices would be easier
                node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.weights_n_bits,
                                                          x.activation_quantization_cfg.activation_n_bits),
                                           reverse=True)

                max_cfg_candidates = n.find_max_candidates_indices()
                assert len(max_cfg_candidates) == 1, \
                    f"A maximal config candidate must be defined, but some node have multiple potential maximal candidates"
                max_candidate_idx = max_cfg_candidates[0]

                activation_quantizers = [ConfigurableActivationQuantizer(**{'node_q_cfg': node_q_cfg_candidates,
                                                                            'max_candidate_idx': max_candidate_idx})] \
                                        * num_of_outputs
            else:
                node_act_qc = n.get_unique_activation_candidates()
                assert len(node_act_qc) == 1, f"Expecting node {n.name} to have a unique activation configuration, " \
                                              f"but {len(node_act_qc)} different configurations exist."
                quantizer_for_node = get_inferable_quantizer_class(QuantizationTarget.Activation,
                                                                   node_act_qc[0].activation_quantization_cfg.activation_quantization_method,
                                                                   BaseKerasInferableQuantizer)
                kwargs = get_inferable_quantizer_kwargs(node_act_qc[0].activation_quantization_cfg,
                                                        QuantizationTarget.Activation)

                activation_quantizers = [quantizer_for_node(**kwargs)] * num_of_outputs

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return KerasActivationQuantizationHolder(activation_quantizers[0])

        Logger.error(f'KerasActivationQuantizationHolder supports a single quantizer but '  # pragma: no cover
                     f'{len(activation_quantizers)} quantizers were found for node {n}')

    def build_model(self) -> Tuple[Model, UserInformation,
                                   Dict[str, Union[KerasQuantizationWrapper, KerasActivationQuantizationHolder]]]:
        """
        Build a Keras mixed-precision model and return it.
        Used the basic Keras model builder to build the model, and adding a mapping between each configurable node to
        a list of layers (from the new model) that are matching to the node (either KerasQuantizationWrapper or
        KerasActivationQuantizationHolder type layers).
        This mapping is used during mixed precision metric computation to enforce pairs of weights-activation bit-width
        candidates when configuring a model.

        Returns: Mixed-precision Keras model.

        """
        model, user_info = super().build_model()

        # creating a mapping between graph nodes and model's layers for mixed precision configurability
        conf_node2layers = {n.name: self._find_layers_in_model_by_node(n, model.layers)
                            for n in self.graph.get_configurable_sorted_nodes()}

        return model, user_info, conf_node2layers

    @staticmethod
    def _get_weights_quant_layers(n: BaseNode, layers_list: List[Layer]) -> List[KerasQuantizationWrapper]:
        """
        Filters KerasQuantizationWrapper layers from an MP model that are matching to the given graph node.

        Args:
            n: A configurable graph node.
            layers_list: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's weights quantization.

        """
        return [_l for _l in layers_list if isinstance(_l, KerasQuantizationWrapper) and _l.layer.name == n.name]

    @staticmethod
    def _get_activation_quant_layers(n: BaseNode, layers_list: List[Layer]) -> List[KerasActivationQuantizationHolder]:
        """
        Filters KerasActivationQuantizationHolder layers from an MP model that are matching to the given graph node.

        Args:
            n: A configurable graph node.
            layers_list: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's activation quantization.

        """
        return [_l for _l in layers_list if isinstance(_l, KerasActivationQuantizationHolder)
                and (_l.inbound_nodes[0].inbound_layers.name == n.name or
                     (isinstance(_l.inbound_nodes[0].inbound_layers, KerasQuantizationWrapper) and
                      _l.inbound_nodes[0].inbound_layers.layer.name == n.name))]

    def _find_layers_in_model_by_node(self, n: BaseNode, layers_list: List[Layer]) -> \
            List[Union[KerasQuantizationWrapper, KerasActivationQuantizationHolder]]:
        """
        Retries layers from an MP model that are matching to the given graph node, that is, these are either
        KerasQuantizationWrapper layers or KerasActivationQuantizationHolder layers that are responsible for the graph
        configurable model quantization.

        Args:
            n: A configurable graph node.
            layers_list: Mixed precision model layers list.

        Returns: A list of layers that responsible for the node's quantization.

        """
        weights_quant = n.is_weights_quantization_enabled()
        act_quant = n.is_activation_quantization_enabled()

        if weights_quant and not act_quant:
            return self._get_weights_quant_layers(n, layers_list)
        elif not weights_quant and act_quant:
            return self._get_activation_quant_layers(n, layers_list)
        elif weights_quant and act_quant:
            return self._get_weights_quant_layers(n, layers_list) + self._get_activation_quant_layers(n, layers_list)
        else:
            Logger.error(f"Expects node {n.name} to have at either weights or activation quantization configured,"  # pragma: no cover
                         f"but both are disabled.")

