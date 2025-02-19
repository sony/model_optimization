from typing import Dict, Optional

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities, \
    OperatorsSet

from model_compression_toolkit.core.common.quantization.quantization_config import CustomOpsetLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework import \
    FrameworkQuantizationCapabilities, OperationsSetToLayers


class AttachTpcToFramework:

    def __init__(self):
        self._opset2layer = None

        # A mapping that associates each layer type in the operation set (with weight attributes and  a quantization
        # configuration in the target platform model) to its framework-specific attribute name. If not all layer types
        # in the operation set are provided in the mapping,  a DefaultDict should be supplied to handle missing entries.
        self._opset2attr_mapping = None  # Mapping of operation sets to their corresponding framework-specific layers

    def attach(self, tpc_model: TargetPlatformCapabilities,
               custom_opset2layer: Optional[Dict[str, 'CustomOpsetLayers']] = None
               ) -> FrameworkQuantizationCapabilities:
        """
        Attaching a TargetPlatformCapabilities which includes a platform capabilities description to specific
        framework's operators.

        Args:
            tpc_model: a TargetPlatformCapabilities object.
            custom_opset2layer: optional set of custom operator sets which allows to add/override the built-in set
                of framework operator, to define a specific behavior for those operators. This dictionary should map
                an operator set unique name to a pair of: a list of framework operators and an optional
                operator's attributes names mapping.

        Returns: a FrameworkQuantizationCapabilities object.

        """

        tpc = FrameworkQuantizationCapabilities(tpc_model)
        custom_opset2layer = custom_opset2layer if custom_opset2layer is not None else {}
        operator_set = tpc_model.operator_set or ()
        with tpc:
            for opset in operator_set:
                if isinstance(opset, OperatorsSet):  # filter out OperatorsSetConcat
                    if opset.name in custom_opset2layer:
                        custom_opset_layers = custom_opset2layer[opset.name]
                        OperationsSetToLayers(opset.name,
                                              layers=custom_opset_layers.operators,
                                              attr_mapping=custom_opset_layers.attr_mapping)

                    elif opset.name in self._opset2layer:
                        # Note that if the user provided a custom operator set with a name that exists in our
                        # pre-defined set of operator sets, we prioritize the user's custom opset definition
                        layers = self._opset2layer[opset.name]
                        if len(layers) > 0:
                            # If the framework does not define any matching operators to a given operator set name that
                            # appears in the TPC, then we just skip it
                            attr_mapping = self._opset2attr_mapping.get(opset.name)
                            OperationsSetToLayers(opset.name, layers, attr_mapping=attr_mapping)
                    else:
                        Logger.critical(f'{opset.name} is defined in TargetPlatformCapabilities, '
                                        f'but is not defined in the framework set of operators or in the provided '
                                        f'custom operator sets mapping.')

        return tpc

