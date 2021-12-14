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


import tensorflow as tf
# As from Tensorflow 2.6, keras is a separate package and some classes should be imported differently.
if tf.__version__ < "2.6":
    from tensorflow.python.keras.engine.node import Node as KerasNode
else:
    from keras.engine.node import Node as KerasNode


from tensorflow.python.util.object_identity import Reference as TFReference
from typing import List, Tuple

from model_compression_toolkit.common.graph.base_graph import OutTensor
from model_compression_toolkit.common.graph.base_node import BaseNode
from model_compression_toolkit.keras.reader.common import is_node_an_input_layer
from model_compression_toolkit.keras.reader.node_builder import build_node

keras = tf.keras
layers = keras.layers


class ConnectivityHandler(object):
    """
    Class containing all information about connections in a graph: nodes and input/output tensors of each node.
    A connection between two nodes is exist if there is a tensor connecting them.
    Tensors are represented by their names.
    """

    def __init__(self):
        self._nodes2input_tensors = dict()  # Node -> List[Tensor]
        self._input_tensors2nodes = dict()  # Tensor -> List[Node]
        self._nodes2output_tensors = dict()  # Node -> List[Tensor]
        self._output_tensors2nodes = dict()  # Tensor -> Node

    def get_nodes(self) -> List[BaseNode]:
        """
        Returns: List of nodes in the connectivity handler.
        """
        return list(self._nodes2input_tensors.keys())

    def is_tensor_connected(self,
                            tensor: str) -> bool:
        """
        Whether a tensor is an input tensor of a node in the connectivity handler.
        Args:
            tensor: Tensor reference to check whether is an input tensor of a node in the connectivity handler.

        Returns:
            Whether a tensor is an input tensor of a node in the connectivity handler.
        """
        return self._input_tensors2nodes.get(tensor) is not None

    def input_tensor2nodes(self,
                           in_tensor: str) -> List[BaseNode]:
        """
        Returns a list of nodes that have a given tensor in their input tensors.
        Args:
            in_tensor: Tensor's reference which the returning nodes list hold in their input tensors.

        Returns:
            List of nodes that have in_tensor in their input tensors.
        """
        return self._input_tensors2nodes[in_tensor] if in_tensor in self._input_tensors2nodes else []

    def output_tensor2node(self,
                           out_tensor: str) -> KerasNode:
        """
        Returns the node that its output is the tensor reference out_tensor.
        Args:
            out_tensor: Tensor reference which the returning node holds in their output tensors.

        Returns:
            A node that has out_tensor in its output tensors.
        """
        return self._output_tensors2nodes[out_tensor] if out_tensor in self._output_tensors2nodes else None

    def node2input_tensors(self,
                           node: BaseNode) -> List[TFReference]:
        """
        Get a list of input tensors of a node.
        Args:
            node: Node to return its input tensors.

        Returns:
            List of references to input tensors of the given node.
        """
        return self._nodes2input_tensors[node] if node in self._nodes2input_tensors else []

    def node2output_tensors(self,
                            node: BaseNode) -> List[TFReference]:
        """
        Get a list of output tensors of a node.
        Args:
            node: Node to return its output tensors.

        Returns:
            List of references to output tensors of the given node.
        """
        return self._nodes2output_tensors[node] if node in self._nodes2output_tensors else []

    def build_inputs_list(self,
                          model_input_tensors_list: List[TFReference]) -> List[KerasNode]:
        """
        Build a list of input nodes of the model ordered by their model's input order.

        Args:
            model_input_tensors_list: List of references to the model's input tensors.

        Returns:
            List of the model input nodes ordered by their indices.
        """
        input_nodes_list = []
        for it in model_input_tensors_list:  # iterate model inputs list
            for n in self._input_tensors2nodes[it]:
                if is_node_an_input_layer(n):  # verify it is an input layer
                    input_nodes_list.append(n)
        return input_nodes_list

    def build_outputs_list(self,
                           model_output_tensors_list: List[TFReference]) -> List[OutTensor]:
        """
        Build a list of model output nodes. Each node is being mapped to a list of two integers tuples. Each tuple
        contains the information about indices a model output tensor gets. The first index in the tuple is the index
        of the tensor
        relatively to the node output tensors. The second index in the tuple is the index of the tensor
        relatively to the model output tensors.
        Args:
            model_output_tensors_list:  List of references to the model's output tensors.

        Returns:
            List of the model output nodes and their output indices.
        """
        outputs_list = []
        for model_ot in model_output_tensors_list:
            out_node = self._output_tensors2nodes.get(model_ot)  # output node of the model
            # index of tensor relatively to the node output tensors
            node_out_index = self._nodes2output_tensors.get(out_node).index(model_ot)
            outputs_list.append(OutTensor(out_node, node_out_index))
        return outputs_list

    def add_node(self,
                 node: KerasNode,
                 input_tensors_list: List[TFReference],
                 outputs_tensors_list: List[TFReference]):
        """
        Add a node and its input/output tensors to the connectivity handler.
        Args:
            node: Node in the graph.
            input_tensors_list: Node's input tensors references.
            outputs_tensors_list: Node's output tensors references.

        """
        self._nodes2input_tensors[node] = input_tensors_list
        self._nodes2output_tensors[node] = outputs_tensors_list
        for input_t in input_tensors_list:  # update dictionary from a tensor to all nodes having this tensor in
            # their input tensors
            if self._input_tensors2nodes.get(input_t) is None:
                self._input_tensors2nodes[input_t] = [node]
            else:
                self._input_tensors2nodes[input_t].append(node)
        # update dictionary from a tensor to the node its is output tensor.
        # There is only one node that can output this tensor, thus it's not a list
        for output_t in outputs_tensors_list:
            self._output_tensors2nodes[output_t] = node

    def get_edge_indices(self,
                         src_node: BaseNode,
                         dst_node: BaseNode,
                         connecting_tensor: TFReference) -> Tuple[int, int]:
        """
        Get indices of an edge by its source/destination nodes and the connecting tensor which defines the edge.
        The indices are the relative positions of the connecting tensor regard input/output tensors of
        destination/source nodes.
        Args:
            src_node: Source node of the edge.
            dst_node: Destination node of the edge.
            connecting_tensor: Tensor reference that connects the source and destination nodes.

        Returns:
            Tuple of two edge indices: source index and destination index.
        """
        src_index = self.node2output_tensors(src_node).index(connecting_tensor)
        dst_index = self.node2input_tensors(dst_node).index(connecting_tensor)
        return src_index, dst_index

    def get_out_edges_params_list(self,
                                  src_node: BaseNode) -> List[tuple]:
        """
        Compute for a given node, all parameters of its outgoing edges.
        Args:
            src_node: Node to consider as source node to compute outgoing edges parameters.

        Returns:
            List of tuples. Each tuple contains parameters of the outgoing edge of src_node: destination node,
            source index, destination index.
        """
        out_edges_params_list = []
        out_tensors = self.node2output_tensors(src_node)
        for ot in out_tensors:  # Loop over output tensors
            if self.is_tensor_connected(ot):
                for dst_node in self.input_tensor2nodes(ot):  # Loop over dst nodes
                    if src_node is not dst_node:  # To handle input layer due to equality of input and output tensors
                        # Get edge indices for the connecting tensor
                        src_index, dst_index = self.get_edge_indices(src_node, dst_node, ot)
                        out_edges_params_list.append((dst_node, src_index, dst_index))

        return out_edges_params_list

    def convert_to_internal_nodes(self):
        """
        Convert nodes in the connectivity handler from a Keras node to an internal node.
        Each internal node is created from a Keras node the connectivity handler has.
        """

        keras_nodes = self.get_nodes()
        node_name_to_node = dict()  # dictionary from node name to node for handling reused layers
        keras_node_to_internal_node = dict()
        for node in keras_nodes:
            internal_node = build_node(node, node_name_to_node)  # build node in the graph
            node_name_to_node[internal_node.name] = internal_node  # update nodes dictionary
            keras_node_to_internal_node[node] = internal_node

        # update _nodes2input_tensors and _nodes2output_tensors with new converted nodes
        for keras_node, internal_node in keras_node_to_internal_node.items():
            self._nodes2input_tensors[internal_node] = self.node2input_tensors(keras_node)
            self._nodes2input_tensors.pop(keras_node)
            self._nodes2output_tensors[internal_node] = self.node2output_tensors(keras_node)
            self._nodes2output_tensors.pop(keras_node)

        # update _input_tensors2nodes
        for it, keras_nodes_list in self._input_tensors2nodes.items():
            self._input_tensors2nodes[it] = [keras_node_to_internal_node[k_node] for k_node in keras_nodes_list]

        # update _output_tensors2nodes
        for ot, mapped_keras_node in self._output_tensors2nodes.items():
            self._output_tensors2nodes[ot] = keras_node_to_internal_node[mapped_keras_node]
