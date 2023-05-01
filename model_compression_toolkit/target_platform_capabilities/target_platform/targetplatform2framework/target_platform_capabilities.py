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
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.operations_to_layers import \
    OperationsToLayers, OperationsSetToLayers
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.target_platform_capabilities_component import TargetPlatformCapabilitiesComponent
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.layer_filter_params import LayerFilterParams
from model_compression_toolkit.target_platform_capabilities.immutable import ImmutableClass
from model_compression_toolkit.target_platform_capabilities.target_platform.op_quantization_config import QuantizationConfigOptions, \
    OpQuantizationConfig
from model_compression_toolkit.target_platform_capabilities.target_platform.operators import OperatorsSetBase
from model_compression_toolkit.target_platform_capabilities.target_platform.target_platform_model import TargetPlatformModel
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework.current_tpc import _current_tpc


class TargetPlatformCapabilities(ImmutableClass):
    """
    Attach framework information to a modeled hardware.
    """
    def __init__(self,
                 tp_model: TargetPlatformModel,
                 name: str = "base",
                 version: str = None):
        """

        Args:
            tp_model (TargetPlatformModel): Modeled hardware to attach framework information to.
            name (str): Name of the TargetPlatformCapabilities.
            version (str): TPC version.
        """

        super().__init__()
        self.name = name
        assert isinstance(tp_model, TargetPlatformModel), f'Target platform model that was passed to TargetPlatformCapabilities must be of type TargetPlatformModel, but has type of {type(tp_model)}'
        self.tp_model = tp_model
        self.op_sets_to_layers = OperationsToLayers() # Init an empty OperationsToLayers
        self.layer2qco, self.filterlayer2qco = {}, {} # Init empty mappings from layers/LayerFilterParams to QC options
        # Track the unused opsets for warning purposes.
        self.__tp_model_opsets_not_used = [s.name for s in tp_model.operator_set]
        self.remove_fusing_names_from_not_used_list()
        self.version = version

    def get_layers_by_opset_name(self, opset_name: str) -> List[Any]:
        """
        Get a list of layers that are attached to an OperatorsSet by the OperatorsSet name.

        Args:
            opset_name: OperatorsSet name to get its layers.


        Returns:
            List of layers/LayerFilterParams that are attached to the opset name.
        """
        opset = self.tp_model.get_opset_by_name(opset_name)
        if opset is None:
            Logger.warning(f'{opset_name} was not found in TargetPlatformCapabilities.')
            return None
        return self.get_layers_by_opset(opset)

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
        for p in self.tp_model.fusing_patterns:
            ops = [self.get_layers_by_opset(x) for x in p.operator_groups_list]
            res.extend(itertools.product(*ops))
        return [list(x) for x in res]


    def get_info(self) -> Dict[str, Any]:
        """

        Returns: Summarization of information in the TargetPlatformCapabilities.

        """
        return {"Target Platform Capabilities": self.name,
                "Version": self.version,
                "Target Platform Model": self.tp_model.get_info(),
                "Operations to layers": {op2layer.name:[l.__name__ for l in op2layer.layers] for op2layer in self.op_sets_to_layers.op_sets_to_layers}}

    def show(self):
        """

        Display the TargetPlatformCapabilities.

        """
        pprint.pprint(self.get_info(), sort_dicts=False, width=110)

    def append_component(self, tpc_component: TargetPlatformCapabilitiesComponent):
        """
        Append a Component (like OperationsSetToLayers) to the TargetPlatformCapabilities.

        Args:
            tpc_component: Component to append to TargetPlatformCapabilities.

        """
        if isinstance(tpc_component, OperationsSetToLayers):
            self.op_sets_to_layers += tpc_component
        else:
            Logger.error(f'Trying to append an unfamiliar TargetPlatformCapabilitiesComponent of type: '
                         f'{type(tpc_component)}')  # pragma: no cover

    def __enter__(self):
        """
        Init a TargetPlatformCapabilities object.
        """
        _current_tpc.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Finalize a TargetPlatformCapabilities object.
        """
        if exc_value is not None:
            print(exc_value, exc_value.args)
            raise exc_value
        self.raise_warnings()
        self.layer2qco, self.filterlayer2qco = self._get_config_options_mapping()
        _current_tpc.reset()
        self.initialized_done()
        return self

    def get_default_op_qc(self) -> OpQuantizationConfig:
        """

        Returns: The default OpQuantizationConfig of the TargetPlatformModel that is attached
        to the TargetPlatformCapabilities.

        """
        return self.tp_model.get_default_op_quantization_config()


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
                qco = self.tp_model.get_config_options_by_operators_set(op2layers.name)
                if qco is None:
                    qco = self.tp_model.default_qco
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
        for f in self.tp_model.fusing_patterns:
            for s in f.operator_groups_list:
                self.remove_opset_from_not_used_list(s.name)

    def remove_opset_from_not_used_list(self,
                                        opset_to_remove: str):
        """
        Remove OperatorsSet name from the unused op list.

        Args:
            opset_to_remove: OperatorsSet name to remove.

        """
        if opset_to_remove in self.__tp_model_opsets_not_used:
            self.__tp_model_opsets_not_used.remove(opset_to_remove)

    def raise_warnings(self):
        """

        Log warnings regards unused opsets.

        """
        for op in self.__tp_model_opsets_not_used:
            Logger.warning(f'{op} is defined in TargetPlatformModel, but is not used in TargetPlatformCapabilities.')
