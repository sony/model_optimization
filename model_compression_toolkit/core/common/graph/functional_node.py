from typing import Dict, Any, Tuple, Type, List, Union

from model_compression_toolkit.verify_packages import FOUND_TF
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
                 op_call_args: Tuple[Any] = None,
                 op_call_kwargs: Dict[str, Any] = None,
                 reuse: bool = False,
                 reuse_group: str = None,
                 quantization_attr: Dict[str, Any] = None,
                 functional_op: Any = None,
                 inputs_as_list: bool = False,
                 has_activation: bool = True,
                 tensor_input_allocs: List[Union[int, str]] = None):
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
            tensor_input_allocs: A list of indices and strings for allocations input tensors in the node's args and kwargs.
        """

        super().__init__(name,
                         framework_attr,
                         input_shape,
                         output_shape,
                         weights,
                         layer_class,
                         reuse,
                         reuse_group,
                         inputs_as_list,
                         quantization_attr,
                         has_activation=has_activation)

        self.op_call_kwargs = op_call_kwargs
        self.op_call_args = list(op_call_args)
        self.functional_op = functional_op
        self.tensor_input_allocs = [] if tensor_input_allocs is None else tensor_input_allocs

    @property
    def type(self):
        """
        A function to get the node's function op for convenient comparison (instead of the layer_class)
        :return: the node's functional_op
        """
        return self.functional_op

    def is_match_type(self, _type: Type) -> bool:
        """
        Check if input type matches the node type, either in instance type or in type name. Checking the
        name string is required because of function types changes that occurred in TF 2.15, because it
        changes the "function" attribute object (e.g. a different tf.add function that will fail the
        equal operation).

        Args:
            _type: other node type
        Returns:
            Whether _type matches the self node type

        """
        names_match = _type.__name__ == self.type.__name__ if FOUND_TF else False
        return super().is_match_type(_type) or names_match
