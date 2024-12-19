from typing import Dict, Tuple, List, Any, Optional

from model_compression_toolkit import DefaultDict
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformModel
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities, \
    OperationsSetToLayers


class TpcAttach2Fw:

    def __init__(self):
        self._opset2layer = None

        # we provide attributes mapping that maps each layer type in the operations set
        # that has weights attributes with provided quantization config (in the tp model) to
        # its framework-specific attribute name.
        # note that a DefaultDict should be provided if not all the layer types in the
        # operation set are provided separately in the mapping.
        self._opset2attr_mapping = None

    def attach(self, tpc_model: TargetPlatformModel,
               custom_opset2layer: Dict[str, Tuple[List[Any], Optional[Dict[str, DefaultDict]]]] = None
               ) -> TargetPlatformCapabilities:

        tpc = TargetPlatformCapabilities(tpc_model)

        with tpc:
            for opset_name, operators in self._opset2layer.items():
                attr_mapping = self._opset2attr_mapping.get(opset_name)
                if attr_mapping is None:
                    OperationsSetToLayers(opset_name, operators)
                else:
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

