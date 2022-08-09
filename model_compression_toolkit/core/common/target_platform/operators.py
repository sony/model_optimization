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
from typing import Dict, Any

from model_compression_toolkit.core.common.target_platform.target_platform_model_component import TargetPlatformModelComponent
from model_compression_toolkit.core.common.target_platform.current_tp_model import _current_tp_model
from model_compression_toolkit.core.common.target_platform.op_quantization_config import QuantizationConfigOptions


class OperatorsSetBase(TargetPlatformModelComponent):
    """
    Base class to represent a set of operators.
    """
    def __init__(self, name: str):
        """

        Args:
            name: Name of OperatorsSet.
        """
        super().__init__(name=name)


class OperatorsSet(OperatorsSetBase):
    def __init__(self,
                 name: str,
                 qc_options: QuantizationConfigOptions = None):
        """
        Set of operators that are represented by a unique label.

        Args:
            name (str): Set's label (must be unique in a TargetPlatformModel).
            qc_options (QuantizationConfigOptions): Configuration options to use for this set of operations.
        """

        super().__init__(name)
        self.qc_options = qc_options
        is_fusing_set = qc_options is None
        self.is_default = _current_tp_model.get().default_qco == self.qc_options or is_fusing_set


    def get_info(self) -> Dict[str,Any]:
        """

        Returns: Info about the set as a dictionary.

        """
        return {"name": self.name,
                "is_default_qc": self.is_default}


class OperatorSetConcat(OperatorsSetBase):
    """
    Concatenate a list of operator sets to treat them similarly in different places (like fusing).
    """
    def __init__(self, *opsets: OperatorsSet):
        """
        Group a list of operation sets.

        Args:
            *opsets (OperatorsSet): List of operator sets to group.
        """
        name = "_".join([a.name for a in opsets])
        super().__init__(name=name)
        self.op_set_list = opsets
        self.qc_options = None  # Concat have no qc options

    def get_info(self) -> Dict[str,Any]:
        """

        Returns: Info about the sets group as a dictionary.

        """
        return {"name": self.name,
                "ops_set_list": [s.name for s in self.op_set_list]}
