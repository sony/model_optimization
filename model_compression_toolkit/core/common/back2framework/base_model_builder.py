from typing import Any, Tuple

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.user_info import UserInformation


class BaseModelBuilder:
    """
    Base class for model builder.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes of graph to append to model's output.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        self.graph = graph
        self.append2output = append2output
        self.return_float_outputs = return_float_outputs

    def build_model(self) -> Tuple[Any, UserInformation]:
        """

        Returns: A framework's model built from its graph.

        """
        raise NotImplemented(f'{self.__class__.__name__} have to implement build_model method.')
