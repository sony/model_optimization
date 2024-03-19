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


from typing import Any, List, Union

from model_compression_toolkit.target_platform_capabilities.target_platform.operators import OperatorSetConcat, \
    OperatorsSet
from model_compression_toolkit.target_platform_capabilities.target_platform.target_platform_model_component import TargetPlatformModelComponent


class Fusing(TargetPlatformModelComponent):
    """
     Fusing defines a list of operators that should be combined and treated as a single operator,
     hence no quantization is applied between them.
    """

    def __init__(self,
                 operator_groups_list: List[Union[OperatorsSet, OperatorSetConcat]],
                 name: str = None):
        """
        Args:
            operator_groups_list (List[Union[OperatorsSet, OperatorSetConcat]]): A list of operator groups, each being either an OperatorSetConcat or an OperatorsSet.
            name (str): The name for the Fusing instance. If not provided, it's generated from the operator groups' names.
        """
        assert isinstance(operator_groups_list,
                          list), f'List of operator groups should be of type list but is {type(operator_groups_list)}'
        assert len(operator_groups_list) >= 2, f'Fusing can not be created for a single operators group'

        # Generate a name from the operator groups if no name is provided
        if name is None:
            name = '_'.join([x.name for x in operator_groups_list])

        super().__init__(name)
        self.operator_groups_list = operator_groups_list

    def contains(self, other: Any) -> bool:
        """
        Determines if the current Fusing instance contains another Fusing instance.

        Args:
            other: The other Fusing instance to check against.

        Returns:
            A boolean indicating whether the other instance is contained within this one.
        """
        if not isinstance(other, Fusing):
            return False

        # Check for containment by comparing operator groups
        for i in range(len(self.operator_groups_list) - len(other.operator_groups_list) + 1):
            for j in range(len(other.operator_groups_list)):
                if self.operator_groups_list[i + j] != other.operator_groups_list[j] and not (
                        isinstance(self.operator_groups_list[i + j], OperatorSetConcat) and (
                        other.operator_groups_list[j] in self.operator_groups_list[i + j].op_set_list)):
                    break
            else:
                # If all checks pass, the other Fusing instance is contained
                return True
        # Other Fusing instance is not contained
        return False

    def get_info(self):
        """
        Retrieves information about the Fusing instance, including its name and the sequence of operator groups.

        Returns:
            A dictionary with the Fusing instance's name as the key and the sequence of operator groups as the value,
            or just the sequence of operator groups if no name is set.
        """
        if self.name is not None:
            return {self.name: ' -> '.join([x.name for x in self.operator_groups_list])}
        return ' -> '.join([x.name for x in self.operator_groups_list])