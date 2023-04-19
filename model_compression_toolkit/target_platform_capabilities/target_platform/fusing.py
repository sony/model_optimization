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


from typing import Any

from model_compression_toolkit.target_platform_capabilities.target_platform.operators import OperatorSetConcat
from model_compression_toolkit.target_platform_capabilities.target_platform.target_platform_model_component import TargetPlatformModelComponent


class Fusing(TargetPlatformModelComponent):

    def __init__(self, operator_groups_list, name=None):
        assert isinstance(operator_groups_list,
                          list), f'List of operator groups should be of type list but is {type(operator_groups_list)}'
        assert len(operator_groups_list) >= 2, f'Fusing can not be created for a single operators group'
        if name is None:
            name = '_'.join([x.name for x in operator_groups_list])
        super().__init__(name)
        self.operator_groups_list = operator_groups_list

    def contains(self, other: Any):
        if not isinstance(other, Fusing):
            return False
        for i in range(len(self.operator_groups_list) - len(other.operator_groups_list) + 1):
            for j in range(len(other.operator_groups_list)):
                if self.operator_groups_list[i + j] != other.operator_groups_list[j] and not (isinstance(self.operator_groups_list[i + j], OperatorSetConcat) and (other.operator_groups_list[j] in self.operator_groups_list[i + j].op_set_list)):
                    break
            else:
                return True
        return False


    def get_info(self):
        if self.name is not None:
            return {self.name: ' -> '.join([x.name for x in self.operator_groups_list])}
        return ' -> '.join([x.name for x in self.operator_groups_list])

