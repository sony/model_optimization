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
from typing import List, Tuple, Any, Dict, Union

import tensorflow as tf
# import tensorflow_model_optimization.quantization.keras.graph_transformations.model_transformer as mt
from keras.engine.base_layer import Layer
from keras.models import Model
from mct_quantizers import KerasQuantizationWrapper, KerasActivationQuantizationHolder, QuantizationTarget
from mct_quantizers.common.get_quantizers import get_inferable_quantizer_class
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder, \
    is_layer_fake_quant, get_node_name_from_layer
from model_compression_toolkit.core.keras.mixed_precision.configurable_activation_quantizer import \
    ConfigurableActivationQuantizer
from model_compression_toolkit.core.keras.mixed_precision.configurable_weights_quantizer import \
    ConfigurableWeightsQuantizer
from model_compression_toolkit.core.keras.quantizer.input_layer_quantize_transform import \
    InputLayerWrapperTransform

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
from model_compression_toolkit.core.keras.quantizer.mixed_precision.selective_quantize_config import \
    SelectiveQuantizeConfig
from packaging import version

from model_compression_toolkit.exporter.model_wrapper.keras.builder.node_to_quantizer import \
    get_inferable_quantizer_kwargs

if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.python.keras.layers.core import TFOpLambda, SlicingOpLambda  # pragma: no cover
else:
    from keras.layers.core import TFOpLambda, SlicingOpLambda

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from model_compression_toolkit.logger import Logger
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
        A function which takes a computational graph node and a keras layer and perform the quantization wrapping.

        Args:
            n: A node of mct graph.
            layer: A keras layer

        Returns: Wrapped layer if the layer should be wrap, otherwise returns the layer as is.

        """

        weights_conf_nodes_names = [n.name for n in self.graph.get_weights_configurable_nodes()]

        if n.is_weights_quantization_enabled():
            kernel_attributes = self.fw_info.get_kernel_op_attributes(n.type)
            if n.name in weights_conf_nodes_names:
                # if len(kernel_attribute) == 0:
                #     Logger.error(f"Node {n.name} has no valid kernel attribute to quantize.")

                return KerasQuantizationWrapper(layer,
                                                weights_quantizers={attr: ConfigurableWeightsQuantizer(
                                                    **self._get_weights_configurable_quantizer_kwargs(n, attr))
                                                                    for attr in kernel_attributes})
            else:
                node_weights_qc = n.get_unique_weights_candidates()
                assert len(node_weights_qc) == 1, f"Expecting node {n.name} to have a unique weights configuration, " \
                                                  f"but {len(node_weights_qc)} different configurations exist."

                quantier_for_node = get_inferable_quantizer_class(QuantizationTarget.Weights,
                                                                  node_weights_qc[0].weights_quantization_cfg.weights_quantization_method,
                                                                  BaseKerasInferableQuantizer)
                kwargs = get_inferable_quantizer_kwargs(n, QuantizationTarget.Weights)

                return KerasQuantizationWrapper(layer,
                                                weights_quantizers={attr: quantier_for_node(**kwargs)
                                                                    for attr in kernel_attributes})

        return layer

    def _get_weights_configurable_quantizer_kwargs(self, n: BaseNode, attr: str) -> Dict[str, Any]:
        """
        Get the quantization parameters for an inferable quantizer.
        Args:
            node: The node for which the quantizer is being created.
            quantization_target: The target of the quantization (weights or activations).
        Returns:
            The quantization parameters as a dictionary.
        """

        assert n.candidates_quantization_cfg is not None
        node_q_cfg_candidates = n.candidates_quantization_cfg
        # sort by descending bit width so using indices would be easier
        node_q_cfg_candidates.sort(key=lambda x: (x.weights_quantization_cfg.weights_n_bits,
                                                  x.activation_quantization_cfg.activation_n_bits), reverse=True)

        # float_weights = [n.get_weights_by_keys(attr) for attr in self.fw_info.get_kernel_op_attributes(n.type)]
        float_weights = n.get_weights_by_keys(attr)

        max_cfg_candidates = n.find_max_candidates_indices()
        assert len(max_cfg_candidates) == 1, \
            f"A maximal config candidate must be defined, but some node have multiple potential maximal candidates"
        max_candidate_idx = max_cfg_candidates[0]

        return {'node_q_cfg': node_q_cfg_candidates,
                'float_weights': float_weights,
                'weight_attrs': attr,
                'max_candidate_idx': max_candidate_idx
                }

    def mixed_precision_activation_holder(self, n):
        """
        Retrieve a KerasActivationQuantizationHolder layer to use for activation quantization for a node.
        If the layer is not supposed to be wrapped with activation quantizers - return None.

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
                assert n.candidates_quantization_cfg is not None
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
                kwargs = get_inferable_quantizer_kwargs(n, QuantizationTarget.Activation)

                activation_quantizers = [quantizer_for_node(**kwargs)] * num_of_outputs

        # Holder by definition uses a single quantizer for the activation quantization
        # thus we make sure this is the only possible case (unless it's a node with no activation
        # quantization, which in this case has an empty list).
        if len(activation_quantizers) == 1:
            return KerasActivationQuantizationHolder(activation_quantizers[0])

        Logger.error(
            f'KerasActivationQuantizationHolder supports a single quantizer but {len(activation_quantizers)} quantizers '
            f'were found for node {n}')

    # def _quantize_node_activations(self,
    #                                node: BaseNode,
    #                                input_tensors: List[TFReference]) -> List[TFReference]:
    #     """
    #     Quantize node's activation given input tensors.
    #
    #     Args:
    #         node: Node to quantize its outputs.
    #         input_tensors: Input tensors of the node.
    #
    #     Returns:
    #         Output of the node.
    #
    #     """
    #     if node.is_all_activation_candidates_equal():
    #         # otherwise, we want to use the float tensor when building the model for MP search
    #         return node.candidates_quantization_cfg[0].activation_quantization_cfg.quantize_node_output(input_tensors)
    #     return input_tensors
    #
    def build_model(self) -> Tuple[Model, UserInformation, Dict[str, Layer]]:
        """
        Build a Keras mixed-precision model and return it.
        Returns: Mixed-precision Keras model.

        """
        model, user_info = super().build_model()

        # creating a mapping between graph nodes and model's layers for mixed precision configurability
        conf_node2layers = {n.name: self._find_layers_in_model_by_node(n, model.layers)
                            for n in self.graph.get_configurable_sorted_nodes()}

        return model, user_info, conf_node2layers

    #
    #     conf_nodes_names = self.graph.get_configurable_sorted_nodes_names()
    #
    #     def _quantize_multiple_nbits(layer):
    #         node = self.oh.layer_to_node_dict.get(layer)
    #         if node is not None:
    #             # Wrap only if its weights should be quantized
    #             if node.name in conf_nodes_names:
    #                 # TODO: Maybe FullyQuantizedQuantizeWrapper to allow using TFOpLambda in MP
    #                 if node.layer_class in [TFOpLambda, SlicingOpLambda]:
    #                     Logger.critical(f"Activation mixed-precision is not supported for layers of type "  # pragma: no cover
    #                                     f"{node.layer_class}. Please modify the TargetPlatformModel object, "
    #                                     f"such that layers of type {node.layer_class} "
    #                                     f"won't have more than one quantization configuration option.")
    #                 return QuantizeWrapper(layer, quantization_config_builder_mixed_precision(node))
    #             return layer
    #
    #         elif is_layer_fake_quant(layer):
    #             return layer
    #         else:
    #             raise Exception(  # pragma: no cover
    #                 f'Mismatch between keras model and graph cant find node named: '
    #                 f'{get_node_name_from_layer(layer)}')
    #
    #     # clone each layer in the model and apply _quantize to the layer.
    #     model = tf.keras.models.clone_model(model,
    #                                         input_tensors=None,
    #                                         clone_function=_quantize_multiple_nbits)
    #
    #     # We use a model transformer to wrap the input layer with QuantizeWrapper,
    #     # to allow layer configuration to different bitwidths.
    #     # A model transformer allows to modify a layer in an existing model, by applying the given list of
    #     # transformers on the model (in this case,
    #     # we only apply single transformer - InputLayerQuantizeTransform)
    #     model_inputs = self.graph.get_inputs()
    #
    #     input_transformer = mt.ModelTransformer(model, [InputLayerWrapperTransform(inp,
    #                                                                                quantization_config_builder_mixed_precision(inp),
    #                                                                                self.get_custom_objects(),
    #                                                                                QuantizeWrapper)
    #                                                     for inp in model_inputs])
    #     model = input_transformer.transform()[0]
    #
    #     return model, user_info

    # @staticmethod
    # def get_custom_objects() -> Dict[str, Any]:
    #     """
    #
    #     Returns: Dictionary of custom objects needed to load this model builder's output.
    #
    #     """
    #     return {QuantizeWrapper.__name__:QuantizeWrapper,
    #             SelectiveQuantizeConfig.__name__: SelectiveQuantizeConfig}

    @staticmethod
    def _get_weights_quant_layers(node, layers_list):
        return [_l for _l in layers_list if isinstance(_l, KerasQuantizationWrapper) and _l.layer.name == node.name]

    @staticmethod
    def _get_activation_quant_layers(node, layers_list):
        return [_l for _l in layers_list if isinstance(_l, KerasActivationQuantizationHolder)
                and _l.input.name == node.name]

    def _find_layers_in_model_by_node(self, n, layers):
        weights_quant = n.is_weights_quantization_enabled()
        act_quant = n.is_activation_quantization_enabled()

        if weights_quant and not act_quant:
            return self._get_weights_quant_layers(n, layers)
        elif not weights_quant and act_quant:
            return self._get_activation_quant_layers(n, layers)
        elif weights_quant and act_quant:
            return self._get_weights_quant_layers(n, layers) + self._get_activation_quant_layers(n, layers)
        else:
            Logger.error(f"Expects node {n.name} to have at either weights or activation quantization configured,"
                         f"but both are disabled.")

