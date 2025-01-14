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

from typing import List, Any, Dict

from model_compression_toolkit.logger import Logger
from model_compression_toolkit.target_platform_capabilities.schema.schema_functions import \
    get_config_options_by_operators_set, is_opset_in_model
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.current_tpc import  _current_tpc
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.framework_quantization_capabilities_component import FrameworkQuantizationCapabilitiesComponent
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import OperatorsSetBase, OperatorSetGroup
from model_compression_toolkit import DefaultDict


class OperationsSetToLayers(FrameworkQuantizationCapabilitiesComponent):
    """
    Associate an OperatorsSet to a list of framework's layers.
    """
    def __init__(self,
                 op_set_name: str,
                 layers: List[Any],
                 attr_mapping: Dict[str, DefaultDict] = None):
        """

        Args:
            op_set_name (str): Name of OperatorsSet to associate with layers.
            layers (List[Any]): List of layers/FilterLayerParams to associate with OperatorsSet.
            attr_mapping (Dict[str, DefaultDict]): A mapping between a general attribute name to a DefaultDict that maps a layer type to the layer's framework name of this attribute.

        """
        self.layers = layers
        self.attr_mapping = attr_mapping
        super(OperationsSetToLayers, self).__init__(name=op_set_name)
        _current_tpc.get().remove_opset_from_not_used_list(op_set_name)

    def __repr__(self) -> str:
        """

        Returns: String to represent the mapping from an OperatorsSet's label to the list of layers.

        """
        return f'{self.name} -> {[x.__name__ for x in self.layers]}'



class OperationsToLayers:
    """
    Gather multiple OperationsSetToLayers to represent mapping of framework's layers to TargetPlatformCapabilities OperatorsSet.
    """
    def __init__(self,
                 op_sets_to_layers: List[OperationsSetToLayers]=None):
        """

        Args:
            op_sets_to_layers (List[OperationsSetToLayers]): List of OperationsSetToLayers where each of them maps an OperatorsSet name to a list of layers that represents the OperatorsSet.
        """
        if op_sets_to_layers is None:  # no mapping was added yet
            op_sets_to_layers = []
        else:
            assert isinstance(op_sets_to_layers, list)
        self.op_sets_to_layers = op_sets_to_layers
        self.validate_op_sets()

    def get_layers_by_op(self,
                         op: OperatorsSetBase) -> Any:
        """
        Get list of layers that are associated with an OperatorsSet object.
        If op is not in OperationsToLayers - return an empty list.

        Args:
            op: OperatorsSetBase object to get its layers.

        Returns:
            List of Layers that are associated with the passed OperatorsSet object.
        """
        for o in self.op_sets_to_layers:
            if op.name == o.name:
                return o.layers
        if isinstance(op, OperatorSetGroup):  # If its a concat - return all layers from all OperatorsSets that in the OperatorSetGroup
            layers = []
            for o in op.operators_set:
                layers.extend(self.get_layers_by_op(o))
            return layers
        Logger.warning(f'{op.name} is not in model.')
        return []

    def get_layers(self) -> Any:
        """
        Get list of layers of all OperatorsSet objects.

        Returns:
            List of Layers that are associated with the passed OperatorsSet object.
        """
        layers = []
        for o in self.op_sets_to_layers:
            layers.extend(o.layers)
        return layers

    def __add__(self,
                op_set_to_layers: OperationsSetToLayers):
        """
        Add a OperationsSetToLayers to self's OperationsSetToLayers existing OperationsSetToLayers objects.
        Args:
            op_set_to_layers: OperationsSetToLayers to add.

        Returns:
            A new OperationsToLayers with the new added OperationsSetToLayers.
        """

        assert isinstance(op_set_to_layers, OperationsSetToLayers)
        new_ops2layers = OperationsToLayers(self.op_sets_to_layers + [op_set_to_layers])
        new_ops2layers.validate_op_sets()
        return new_ops2layers

    def validate_op_sets(self):
        """

        Validate there are no violations in OperationsToLayers.

        """
        existing_layers = {}  # Keep tracking layers in OperationsToLayers
        existing_opset_names = []  # Keep tracking OperatorsSet names

        for ops2layers in self.op_sets_to_layers:
            assert isinstance(ops2layers,
                              OperationsSetToLayers), f'Operators set should be of type OperationsSetToLayers but it ' \
                                                      f'is of type {type(ops2layers)}'

            # Assert that opset has a unique name.
            assert ops2layers.name not in existing_opset_names, f'OperationsSetToLayers names should be unique, but {ops2layers.name} appears to violate it.'
            existing_opset_names.append(ops2layers.name)

            # Assert that a layer does not appear in more than a single OperatorsSet in the TargetPlatformCapabilities.
            for layer in ops2layers.layers:
                qco_by_opset_name = get_config_options_by_operators_set(_current_tpc.get().tpc, ops2layers.name)
                if layer in existing_layers:
                    Logger.critical(f'Found layer {layer.__name__} in more than one '
                                    f'OperatorsSet')  # pragma: no cover
                else:
                    existing_layers.update({layer: qco_by_opset_name})
