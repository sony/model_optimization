# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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

import pprint
from typing import Any, Dict

from model_compression_toolkit.common.target_platform.current_tp_model import _current_tp_model, \
    get_current_tp_model
from model_compression_toolkit.common.target_platform.fusing import Fusing
from model_compression_toolkit.common.target_platform.target_platform_model_component import \
    TargetPlatformModelComponent
from model_compression_toolkit.common.target_platform.op_quantization_config import OpQuantizationConfig, \
    QuantizationConfigOptions
from model_compression_toolkit.common.target_platform.operators import OperatorsSetBase
from model_compression_toolkit.common.immutable import ImmutableClass
from model_compression_toolkit.common.logger import Logger


def get_default_quantization_config_options() -> QuantizationConfigOptions:
    """

    Returns: The default QuantizationConfigOptions of the model. This is the options
    to use when a layer's options is queried and it wasn't specified in the TargetPlatformCapabilities.
    The default QuantizationConfigOptions always contains a single option.

    """
    return get_current_tp_model().default_qco


def get_default_quantization_config():
    """

    Returns: The default OpQuantizationConfig of the model. This is the OpQuantizationConfig
    to use when a layer's options is queried and it wasn't specified in the TargetPlatformCapabilities.
    This OpQuantizationConfig is the single option in the default QuantizationConfigOptions.

    """

    return get_current_tp_model().get_default_op_quantization_config()


class TargetPlatformModel(ImmutableClass):
    """
    Modeling of the hardware the quantized model will use during inference.
    The model contains definition of operators, quantization configurations of them, and
    fusing patterns so that multiple operators will be combined into a single operator.
    """

    def __init__(self,
                 default_qco: QuantizationConfigOptions,
                 name="default_tp_model"):
        """

        Args:
            default_qco (QuantizationConfigOptions): Default QuantizationConfigOptions to use for operators that their QuantizationConfigOptions are not defined in the model.
            name (str): Name of the model.
        """

        super().__init__()
        self.name = name
        self.operator_set = []
        assert isinstance(default_qco, QuantizationConfigOptions)
        assert len(default_qco.quantization_config_list) == 1, \
            f'Default QuantizationConfigOptions must contain only one option'
        self.default_qco = default_qco
        self.fusing_patterns = []

    def get_config_options_by_operators_set(self,
                                            operators_set_name: str) -> QuantizationConfigOptions:
        """
        Get the QuantizationConfigOptions of a OperatorsSet by the OperatorsSet name.
        If the name is not in the model, the default QuantizationConfigOptions is returned.

        Args:
            operators_set_name: Name of OperatorsSet to get.

        Returns:
            QuantizationConfigOptions to use for ops in OperatorsSet named operators_set_name.
        """
        for op_set in self.operator_set:
            if operators_set_name == op_set.name:
                return op_set.qc_options
        return get_default_quantization_config_options()

    def get_default_op_quantization_config(self) -> OpQuantizationConfig:
        """

        Returns: The default OpQuantizationConfig of the TargetPlatformModel.

        """
        assert len(self.default_qco.quantization_config_list) == 1, \
            f'Default quantization configuration options must contain only one option,' \
            f' but found {len(get_current_tp_model().default_qco.quantization_config_list)} configurations.'
        return self.default_qco.quantization_config_list[0]

    def is_opset_in_model(self,
                          opset_name: str) -> bool:
        """
        Check whether an operators set is defined in the model or not.

        Args:
            opset_name: Operators set name to check.

        Returns:
            Whether an operators set is defined in the model or not.
        """
        return opset_name in [x.name for x in self.operator_set]

    def get_opset_by_name(self,
                          opset_name: str) -> OperatorsSetBase:
        """
        Get an OperatorsSet object from the model by its name.
        If name is not in the model - None is returned.

        Args:
            opset_name: OperatorsSet name to retrieve.

        Returns:
            OperatorsSet object with the name opset_name, or None if opset_name is not in the model.
        """

        opset_list = [x for x in self.operator_set if x.name == opset_name]
        assert len(opset_list) <= 1, f'Found more than one OperatorsSet in' \
                                     f' TargetPlatformModel with the name {opset_name}. ' \
                                     f'OperatorsSet name must be unique.'
        if len(opset_list) == 0:  # opset_name is not in the model.
            return None

        return opset_list[0]  # There's one opset with that name

    def append_component(self,
                         tp_model_component: TargetPlatformModelComponent):
        """
        Attach a TargetPlatformModel component to the model. Components can be for example:
        Fusing, OperatorsSet, etc.

        Args:
            tp_model_component: Component to attach to the model.

        """
        if isinstance(tp_model_component, Fusing):
            self.fusing_patterns.append(tp_model_component)
        elif isinstance(tp_model_component, OperatorsSetBase):
            self.operator_set.append(tp_model_component)
        else:
            raise Exception(f'Trying to append an unfamiliar TargetPlatformModelComponent of type: {type(tp_model_component)}')

    def __enter__(self):
        """
        Start defining the TargetPlatformModel using 'with'.

        Returns: Initialized TargetPlatformModel object.

        """
        _current_tp_model.set(self)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """
        Finish defining the TargetPlatformModel at the end of the 'with' clause.
        Returns the final and immutable TargetPlatformModel instance.
        """

        if exc_value is not None:
            print(exc_value, exc_value.args)
            raise exc_value
        self.__validate_model()  # Assert that model is valid.
        _current_tp_model.reset()
        self.initialized_done()  # Make model immutable.
        return self

    def __validate_model(self):
        """

        Assert model is valid.
        Model is invalid if, for example, it contains multiple operator sets with the same name,
        as their names should be unique.

        """
        opsets_names = [op.name for op in self.operator_set]
        if (len(set(opsets_names)) != len(opsets_names)):
            Logger.error(f'OperatorsSet must have unique names')

    def get_default_config(self) -> OpQuantizationConfig:
        """

        Returns:

        """
        assert len(self.default_qco.quantization_config_list) == 1, \
            f'Default quantization configuration options must contain only one option,' \
            f' but found {len(self.default_qco.quantization_config_list)} configurations.'
        return self.default_qco.quantization_config_list[0]

    def get_info(self) -> Dict[str, Any]:
        """

        Returns: Dictionary that summarizes the TargetPlatformModel properties (for display purposes).

        """
        return {"Model name": self.name,
                "Default quantization config": self.get_default_config().get_info(),
                "Operators sets": [o.get_info() for o in self.operator_set],
                "Fusing patterns": [f.get_info() for f in self.fusing_patterns]
                }

    def show(self):
        """

        Display the TargetPlatformModel.

        """
        pprint.pprint(self.get_info(), sort_dicts=False)
