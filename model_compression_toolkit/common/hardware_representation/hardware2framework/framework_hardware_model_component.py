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

from model_compression_toolkit.common.hardware_representation.hardware2framework.current_framework_hardware_model import  _current_framework_hardware_model


class FrameworkHardwareModelComponent:
    def __init__(self, name: str):
        self.name = name
        _current_framework_hardware_model.get().append_component(self)

    def get_info(self):
        return {}
