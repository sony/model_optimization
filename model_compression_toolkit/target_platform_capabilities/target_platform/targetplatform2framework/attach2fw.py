from typing import Dict, Tuple, List, Any, Optional

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities, \
    OperationsSetToLayers


class AttachTpModelToFw:

    def __init__(self):
        self._opset2layer = None

        # A mapping that associates each layer type in the operation set (with weight attributes and  a quantization
        # configuration in the target platform model) to its framework-specific attribute name. If not all layer types
        # in the operation set are provided in the mapping,  a DefaultDict should be supplied to handle missing entries.
        self._opset2attr_mapping = None  # Mapping of operation sets to their corresponding framework-specific layers

    def attach(self, tpc_model: TargetPlatformModel,
               custom_opset2layer: Dict[str, Tuple[List[Any], Optional[Dict[str, DefaultDict]]]] = None
               ) -> TargetPlatformCapabilities:
        """
        Attaching a TargetPlatformModel which includes a platform capabilities description to specific
        framework's operators.

        Args:
            tpc_model: a TargetPlatformModel object.
            custom_opset2layer: optional set of custom operator sets which allows to add/override the built-in set
                of framework operator, to define a specific behavior for those operators. This dictionary should map
                an operator set unique name to a pair of: a list of framework operators and an optional
                operator's attributes names mapping.

        Returns: a TargetPlatformCapabilities object.

        """

        tpc = TargetPlatformCapabilities(tpc_model)

        with tpc:
            for opset_name, operators in self._opset2layer.items():
                attr_mapping = self._opset2attr_mapping.get(opset_name)
                OperationsSetToLayers(opset_name, operators, attr_mapping=attr_mapping)

            if custom_opset2layer is not None:
                for opset_name, operators in custom_opset2layer.items():
                    if len(operators) == 1:
                        OperationsSetToLayers(opset_name, operators[0])
                    elif len(operators) == 2:
                        OperationsSetToLayers(opset_name, operators[0], attr_mapping=operators[1])
                    else:
                        raise ValueError(f"Custom operator set to layer mapping should include up to 2 elements - "
                                         f"a list of layers to attach to the operator and an optional mapping of "
                                         f"attributes names, but given a mapping contains {len(operators)} elements.")

        return tpc

