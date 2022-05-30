from typing import Dict, Any, Tuple, List

from model_compression_toolkit.core.common.graph.base_node import BaseNode
import numpy as np


class FunctionalNode(BaseNode):
    """
    Node that represents function ops with arguments to pass when building back the model.
    """

    def __init__(self,
                 name: str,
                 framework_attr: Dict[str, Any],
                 input_shape: Tuple[Any],
                 output_shape: Tuple[Any],
                 weights: Dict[str, np.ndarray],
                 layer_class: type,
                 op_call_args: List[Any] = None,
                 op_call_kwargs: Dict[str, Any] = None,
                 reuse: bool = False,
                 reuse_group: str = None,
                 quantization_attr: Dict[str, Any] = None,
                 functional_op: Any = None,
                 inputs_as_list: bool = False,
                 has_activation: bool = True):
        """
        Init a FunctionalNode object.

        Args:
            name: Node's name
            framework_attr: Framework attributes the layer had which the node holds.
            input_shape: Input tensor shape of the node.
            output_shape: Input tensor shape of the node.
            weights: Dictionary from a variable name to the weights with that name in the layer the node represents.
            layer_class: Class path of the layer this node represents.
            op_call_args: Arguments list to pass when calling the layer.
            op_call_kwargs: Key-Word Arguments dictionary with values to pass when calling the layer.
            reuse: Whether this node was duplicated and represents a reused layer.
            reuse_group: Name of group of nodes from the same reused layer.
            quantization_attr: Attributes the node holds regarding how it should be quantized.
            functional_op: The op the node implements.
            inputs_as_list: Whether to pass the node its input tensors as a list or not when calling the layer.
            has_activation: Whether the node has activations that we might want to quantize.

        """

        super().__init__(name,
                         framework_attr,
                         input_shape,
                         output_shape,
                         weights,
                         layer_class,
                         reuse,
                         reuse_group,
                         quantization_attr,
                         has_activation=has_activation)

        self.op_call_kwargs = op_call_kwargs
        self.op_call_args = op_call_args
        self.functional_op = functional_op
        self.inputs_as_list = inputs_as_list

    @property
    def type(self):
        """
        A function to get the node's function op for convenient comparison (instead of the layer_class)
        :return: the node's functional_op
        """
        return self.functional_op
