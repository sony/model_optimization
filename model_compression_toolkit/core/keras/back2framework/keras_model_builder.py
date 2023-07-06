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

import tensorflow as tf
from keras.models import Model
from packaging import version

from model_compression_toolkit.core.common.back2framework.base_model_builder import BaseModelBuilder
from model_compression_toolkit.core.common.user_info import UserInformation

# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if version.parse(tf.__version__) < version.parse("2.6"):
    from tensorflow.keras.layers import Input
    from tensorflow.python.keras.layers.core import TFOpLambda
    from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
    from tensorflow.python.keras.layers import Layer
else:
    from keras import Input
    from keras.layers.core import TFOpLambda
    from keras.engine.base_layer import TensorFlowOpLayer, Layer

from typing import Any, Dict, List, Tuple, Callable
from tensorflow.python.util.object_identity import Reference as TFReference
from model_compression_toolkit.core.common.graph.functional_node import FunctionalNode
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.graph.edge import EDGE_SINK_INDEX
from model_compression_toolkit.core.keras.back2framework.instance_builder import OperationHandler
from model_compression_toolkit.core.keras.reader.connectivity_handler import OutTensor

# In tf2.3 fake quant node is implemented as TensorFlowOpLayer, while in tf2.4 as TFOpLambda.
FQ_NODE_OP_V2_3 = 'FakeQuantWithMinMaxVars'
FQ_NODE_OP_V2_4 = 'quantization.fake_quant_with_min_max_vars'
BATCH_INPUT_SHAPE = 'batch_input_shape'


def get_node_name_from_layer(layer: Layer) -> str:
    """
    Get a node's name from the layer it was built from. For TensorFlowOpLayer
    we remove the prefix "tf_op_layer".

    Args:
        layer: Keras Layer to get its corresponding node's name.

    Returns:
        Name of the node that was built from the passed layer.
    """

    name = layer.name
    if isinstance(layer, TensorFlowOpLayer):  # remove TF op layer prefix
        name = '_'.join(name.split('_')[3:])
    return name


def is_layer_fake_quant(layer: Layer) -> bool:
    """
    Check whether a Keras layer is a fake quantization layer or not.
    Args:
        layer: Layer to check if it's a fake quantization layer or not.

    Returns:
        Whether a Keras layer is a fake quantization layer or not.
    """
    # in tf2.3 fake quant node is implemented as TensorFlowOpLayer, while in tf2.4 as TFOpLambda
    return (isinstance(layer, TensorFlowOpLayer) and layer.node_def.op == FQ_NODE_OP_V2_3) or (
            isinstance(layer, TFOpLambda) and layer.symbol == FQ_NODE_OP_V2_4)


class KerasModelBuilder(BaseModelBuilder):
    """
    Builder for Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                 return_float_outputs: bool = False,
                 wrapper: Callable = None,
                 get_activation_quantizer_holder_fn: Callable=None):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
            wrapper: A function wrapper keras Layers.
            get_activation_quantizer_holder_fn: Function to retrieve a quantization holder for a node.

        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

        # Build an OperationHandler to handle conversions from graph nodes to Keras operators.
        self.oh = OperationHandler(self.graph,
                                   wrapper=wrapper)
        self.wrapper = wrapper
        self.get_activation_quantizer_holder = get_activation_quantizer_holder_fn

    @property
    def use_activation_holder_during_model_building(self) -> bool:
        """

        Returns: Whether the model builder uses KerasActivationQuantizationHolder during
        model building (by adding it as a layer when converting the graph to the Keras model)
        or not. If so - the model builder expects the activation quantizers to not be wrapped
        in KerasQuantizeWrapper that was received in its init.

        """
        return self.get_activation_quantizer_holder is not None

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
        raise NotImplemented(f'{self.__class__.__name__} did not implement a method for quantizating '
                             f'activation nodes.')  # pragma: no cover

    def build_model(self) -> Tuple[Model, UserInformation]:
        """
        Build a Keras model and return it.
        Returns: Keras model.

        """
        node_to_output_tensors_dict = dict()
        node_to_output_tensors_dict_float = dict()
        model_output_tensors = []

        # Create a list of output nodes with their tensors' indices that are model's output. When building
        # in float mode, if a node has multiple out tensors, the node should appear in append2output multiple
        # times, thus creating OutTensor for each output tensor.
        if self.append2output is not None:
            output_list = [OutTensor(n, 0) for n in self.append2output]
        else:
            output_list = self.graph.get_outputs()

        # Hold a dictionary from an input node to its corresponding input tensor. It is needed for when
        # building the model. Initially input nodes with input tensors are added to the dictionary,
        # as they're not added later.
        input_nodes_to_input_tensors = {inode: Input(inode.framework_attr[BATCH_INPUT_SHAPE][1:], name=inode.name)
                                        for inode in self.graph.get_inputs()}

        # Build a list of the model's input tensors. Switching from a dictionary to a list
        # to keep the tensors input order, since inputs in Graph are ordered by their indices.
        inputs_list = []
        for input_node in self.graph.get_inputs():
            inputs_list.append(input_nodes_to_input_tensors.get(input_node))

        # Build a dictionary from node to its output tensors, by applying the layers sequentially.
        for n in self.oh.node_sort:
            op_func = self.oh.get_node_op_function(n)  # Get node operation function

            input_tensors = self._build_input_tensors_list(n,
                                                           node_to_output_tensors_dict)  # Fetch Node inputs
            out_tensors_of_n, out_tensors_of_n_float = self._run_operation(n,  # Run node operation and fetch outputs
                                                                           input_tensors,
                                                                           op_func,
                                                                           input_nodes_to_input_tensors)

            if isinstance(out_tensors_of_n, (list, tuple)):
                node_to_output_tensors_dict.update({n: out_tensors_of_n})
                node_to_output_tensors_dict_float.update({n: out_tensors_of_n_float})
            else:
                node_to_output_tensors_dict.update({n: [out_tensors_of_n]})
                node_to_output_tensors_dict_float.update({n: [out_tensors_of_n_float]})

        # convert node_to_output_tensors_dict keys to nodes' names since oh.node_sort
        # contains different objects than original graph nodes.
        node_name_to_outtensors = self._convert_node2name(node_to_output_tensors_dict)
        node_name_to_outtensors_float = self._convert_node2name(node_to_output_tensors_dict_float)

        for ot in output_list:
            if len(node_name_to_outtensors[ot.node.name]) == 1 or self.append2output is None:
                if self.return_float_outputs:
                    model_output_tensors.append(node_name_to_outtensors_float[ot.node.name][ot.node_out_index])
                else:
                    model_output_tensors.append(node_name_to_outtensors[ot.node.name][ot.node_out_index])
            else:  # When building float model - we collect all outputs from all nodes regardless the actual
                # model's outputs
                if self.return_float_outputs:
                    # In case of GPTQ output the float data for the loss and not quantized.
                    model_output_tensors.append(node_name_to_outtensors_float[ot.node.name])
                else:
                    model_output_tensors.append(node_name_to_outtensors[ot.node.name])

        # Build the model.
        model = tf.keras.Model(inputs=inputs_list, outputs=model_output_tensors)

        return model, self.graph.user_info

    def _convert_node2name(self, in_node_to_output_tensors_dict):
        node_name_to_outtensors = dict()
        for node, tensors in in_node_to_output_tensors_dict.items():
            node_name_to_outtensors[node.name] = tensors
        return node_name_to_outtensors

    def _build_input_tensors_list(self,
                                  node: BaseNode,
                                  node_to_output_tensors_dict: Dict[BaseNode, List[TFReference]]) -> List[
        List[TFReference]]:
        """
        Given a node, build a list of input tensors the node gets. The list is built
        based on the node's incoming edges and previous nodes' output tensors.

        Args:
            node: Node to build its input tensors list.
            node_to_output_tensors_dict: A dictionary from a node to its output tensors.

        Returns:
            A list of the node's input tensors.
        """

        input_tensors = []
        # Go over a sorted list of the node's incoming edges, and for each source node get its output tensors.
        # Append them in a result list.
        for ie in self.graph.incoming_edges(node, sort_by_attr=EDGE_SINK_INDEX):
            _input_tensors = [node_to_output_tensors_dict[ie.source_node][ie.source_index]]
            input_tensors.append(_input_tensors)
        return input_tensors

    def _run_operation(self,
                       n: BaseNode,
                       input_tensors: List[List[TFReference]],
                       op_func: Layer,
                       input_nodes_to_input_tensors: Dict[BaseNode, Any]) -> List[TFReference]:
        """
        Applying the layer (op_func) to the input tensors (input_tensors).
        If quantized is set to True, and the layer's corresponding node (n) has quantization
        attributes, an additional fake-quantization node is built and appended to the layer.

        Args:
            n: The corresponding node of the layer it runs.
            input_tensors: List of references to Keras tensors that are the layer's inputs.
            op_func: Layer to apply to the input tensors.
            input_nodes_to_input_tensors: A dictionary from a node to its input tensors.

        Returns:
            A list of references to Keras tensors. The layer's output tensors after applying the
            layer to the input tensors.
        """
        if len(input_tensors) == 0:  # Placeholder handling
            out_tensors_of_n_float = input_nodes_to_input_tensors[n]
            out_tensors_of_n = self._run_operation_activation_quantization(n,
                                                                           out_tensors_of_n_float)
        else:
            input_tensors = [tensor for tensor_list in input_tensors for tensor in tensor_list]  # flat list of lists
            # Build a functional node using its args
            if isinstance(n, FunctionalNode):
                if n.inputs_as_list:  # If the first argument should be a list of tensors:
                    out_tensors_of_n_float = op_func(input_tensors, *n.op_call_args, **n.op_call_kwargs)
                else:  # If the input tensors should not be a list but iterated:
                    out_tensors_of_n_float = op_func(*input_tensors, *n.op_call_args, **n.op_call_kwargs)
            else:
                # If operator expects a single input tensor, it cannot be a list as it should
                # have a dtype field.
                if len(input_tensors) == 1:
                    input_tensors = input_tensors[0]
                out_tensors_of_n_float = op_func(input_tensors)

            out_tensors_of_n = self._run_operation_activation_quantization(n,
                                                                           out_tensors_of_n_float)

        # Save a mapping from the layer that created the tensor to the node (as this layer is not the
        # same instance as op_func. We do this to solve an issue that names are different between these
        # layers, thus we can not rely on the op_func name during model cloning (such as GPTQ, MP, etc.)
        if not n.reuse:
            if isinstance(out_tensors_of_n_float, (list, tuple)):
                # If layer has multiple outputs (e.g., split) we take the layer of the first output (as all outputs are
                # from the same layer).
                layer_from_tensor = out_tensors_of_n_float[0].node.layer
            else:
                layer_from_tensor = out_tensors_of_n_float.node.layer
            self.oh.layer_to_node_dict[layer_from_tensor] = n

        return out_tensors_of_n, out_tensors_of_n_float

    def _run_operation_activation_quantization(self,
                                               node: BaseNode,
                                               node_outputs: List[TFReference]):
        """
        Quantize node's activations

        Args:
            node: Node to quantize its activations
            node_outputs: Output tensors of the float node.

        Returns:
            Quantized node's outputs.
        """
        if self.wrapper is not None:

            # In case the activation quantizer is attached out of the wrapper we use get_activation_quantizer_holder
            # for the activation quantization holder (if the node's activations are quantized)
            if node.is_activation_quantization_enabled() and self.use_activation_holder_during_model_building:
                activation_quantizer_holder = self.get_activation_quantizer_holder(node)
                quantized_node_outputs = activation_quantizer_holder(node_outputs)
                return quantized_node_outputs

        elif node.is_activation_quantization_enabled():  # Used only when old exporter is used
            quantized_node_outputs = self._quantize_node_activations(node,
                                                                     node_outputs)
            return quantized_node_outputs

        return node_outputs
