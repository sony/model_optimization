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


import itertools
import pprint
from typing import List, Any, Dict, Tuple

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set, get_default_op_quantization_config, get_opset_by_name
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.operations_to_layers import OperationsToLayers, \
    OperationsSetToLayers
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities_component import \
    FrameworkQuantizationCapabilitiesComponent
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.layer_filter_params import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.immutable import ImmutableClass
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities, OperatorsSetBase, \
    OpQuantizationConfig, QuantizationConfigOptions
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.current_tpc import  _current_tpc

class FrameworkQuantizationCapabilities(ImmutableClass):
    """
    Attach framework information to a modeled hardware.
    """
    def __init__(self,
                 tpc: TargetPlatformCapabilities,
                 name: str = "base"):
        """

        Args:
            tpc (TargetPlatformCapabilities): Modeled hardware to attach framework information to.
            name (str): Name of the FrameworkQuantizationCapabilities.
        """

        super().__init__()
        self.name = name
        assert isinstance(tpc, TargetPlatformCapabilities), f'Target platform model that was passed to FrameworkQuantizationCapabilities must be of type TargetPlatformCapabilities, but has type of {type(tpc)}'
        self.tpc = tpc
        self.op_sets_to_layers = OperationsToLayers() # Init an empty OperationsToLayers
        self.layer2qco, self.filterlayer2qco = {}, {} # Init empty mappings from layers/LayerFilterParams to QC options
        # Track the unused opsets for warning purposes.
        operator_set = tpc.operator_set or ()
        self.__tpc_opsets_not_used = [s.name for s in operator_set]
        self.remove_fusing_names_from_not_used_list()

    def get_layers_by_opset_name(self, opset_name: str) -> List[Any]:
        """
        Get a list of layers that are attached to an OperatorsSet by the OperatorsSet name.

        Args:
            opset_name: OperatorsSet name to get its layers.


        Returns:
            List of layers/LayerFilterParams that are attached to the opset name.
        """
        opset = get_opset_by_name(self.tpc, opset_name)
        if opset is None:
            Logger.warning(f'{opset_name} was not found in FrameworkQuantizationCapabilities.')
            return None
        return self.get_layers_by_opset(opset)

    def get_layers(self) -> List[Any]:
        """
        Get a list of layers of all OperatorsSet objects.

        Returns:
            List of layers/LayerFilterParams in the TPC.
        """
        return self.op_sets_to_layers.get_layers()

    def get_layers_by_opset(self, op: OperatorsSetBase) -> List[Any]:
        """
        Get a list of layers that are attached to an OperatorsSet by the OperatorsSet object.

        Args:
            op: OperatorsSet object to get its layers.

        Returns:
            List of layers/LayerFilterParams that are attached to the OperatorSet object.
        """
        return self.op_sets_to_layers.get_layers_by_op(op)

    def get_fusing_patterns(self) -> List[List[Any]]:
        """

        Returns: List of patterns of layers/LayerFilterParams to fuse.

        """
        res = []
        if self.tpc.fusing_patterns is None:
            return res
        for p in self.tpc.fusing_patterns:
            ops = [self.get_layers_by_opset(x) for x in p.operator_groups]
            res.extend(itertools.product(*ops))
        return [list(x) for x in res]


    def get_info(self) -> Dict[str, Any]:
        """

        Returns: Summarization of information in the FrameworkQuantizationCapabilities.

        """
        return {"Target Platform Capabilities": self.name,
                "Minor version": self.tpc.tpc_minor_version,
                "Patch version": self.tpc.tpc_patch_version,
                "Platform type": self.tpc.tpc_platform_type,
                "Target Platform Model": self.tpc.get_info(),
                "Operations to layers": {op2layer.name:[l.__name__ for l in op2layer.layers] for op2layer in self.op_sets_to_layers.op_sets_to_layers}}

    def show(self):
        """

        Display the FrameworkQuantizationCapabilities.

        """
        pprint.pprint(self.get_info(), sort_dicts=False, width=110)

    def append_component(self, tpc_component: FrameworkQuantizationCapabilitiesComponent):
        """
        Append a Component (like OperationsSetToLayers) to the FrameworkQuantizationCapabilities.

        Args:
            tpc_component: Component to append to FrameworkQuantizationCapabilities.

        """
        if isinstance(tpc_component, OperationsSetToLayers):
            self.op_sets_to_layers += tpc_component
        else:
            Logger.critical(f"Attempt to append an unrecognized 'FrameworkQuantizationCapabilitiesComponent' of type: '{type(tpc_component)}'. Ensure the component is compatible.")  # pragma: no cover

    def __enter__(self):
        """
        Init a FrameworkQuantizationCapabilities object.
        """
        _current_tpc.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Finalize a FrameworkQuantizationCapabilities object.
        """
        if exc_value is not None:
            print(exc_value, exc_value.args)
            raise exc_value
        self.layer2qco, self.filterlayer2qco = self._get_config_options_mapping()
        _current_tpc.reset()
        self.initialized_done()
        return self

    def get_default_op_qc(self) -> OpQuantizationConfig:
        """

        Returns: The default OpQuantizationConfig of the TargetPlatformCapabilities that is attached
        to the FrameworkQuantizationCapabilities.

        """
        return get_default_op_quantization_config(self.tpc)


    def _get_config_options_mapping(self) -> Tuple[Dict[Any, QuantizationConfigOptions],
                                                   Dict[LayerFilterParams, QuantizationConfigOptions]]:
        """
        Build mapping from layers to their QuantizationConfigOptions (and from LayerFilterParams
        to their QuantizationConfigOptions).

        Returns: Two mappings from layers/LayerFilterParams to their QuantizationConfigOptions.

        """
        layer2qco = {}
        filterlayer2qco = {}
        for op2layers in self.op_sets_to_layers.op_sets_to_layers:
            for l in op2layers.layers:
                qco = get_config_options_by_operators_set(self.tpc, op2layers.name)
                if qco is None:
                    qco = self.tpc.default_qco

                # here, we need to take care of mapping a general attribute name into a framework and
                # layer type specific attribute name.
                # attr_mapping is a mapping between an attribute generic name to a dictionary that maps each
                # layer type to its framework-specific attribute name.
                # in the loop below, v is the inner dictionary.
                layer_attrs_mapping = None if op2layers.attr_mapping is None else \
                    {k: v.get(l) for k, v in op2layers.attr_mapping.items()}
                qco = qco.clone_and_map_weights_attr_keys(layer_attrs_mapping)

                if isinstance(l, LayerFilterParams):
                    filterlayer2qco.update({l: qco})
                else:
                    layer2qco.update({l: qco})
        return layer2qco, filterlayer2qco

    def remove_fusing_names_from_not_used_list(self):
        """
        Remove OperatorSets names from the list of the unused sets (so a warning
        will not be displayed).
        """
        if self.tpc.fusing_patterns is not None:
            for f in self.tpc.fusing_patterns:
                for s in f.operator_groups:
                    self.remove_opset_from_not_used_list(s.name)

    def remove_opset_from_not_used_list(self,
                                        opset_to_remove: str):
        """
        Remove OperatorsSet name from the unused op list.

        Args:
            opset_to_remove: OperatorsSet name to remove.

        """
        if opset_to_remove in self.__tpc_opsets_not_used:
            self.__tpc_opsets_not_used.remove(opset_to_remove)

    @property
    def is_simd_padding(self) -> bool:
        """

        Returns: Check if the TP model defines that padding due to SIMD constrains occurs.

        """
        return self.tpc.is_simd_padding
