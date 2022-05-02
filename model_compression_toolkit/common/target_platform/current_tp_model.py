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

def get_current_model():
    """

    Returns: The current HardwaeModel that is being used and accessed.

    """
    return _current_hardware_model.get()


class CurrentTPModel:
    """
    Wrapper of the current TargetPlatformModel object that is being accessed and defined.
    """

    def __init__(self):
        super(CurrentTPModel, self).__init__()
        self.hwm = None

    def get(self):
        """

        Returns: The current TargetPlatformModel that is being defined.

        """
        if self.hwm is None:
            raise Exception('Hardware model is not initialized.')
        return self.hwm

    def reset(self):
        """

        Reset the current TargetPlatformModel so a new TargetPlatformModel can be wrapped and
        used as the current TargetPlatformModel object.

        """
        self.hwm = None

    def set(self, hwm):
        """
        Set and wrap a TargetPlatformModel as the current TargetPlatformModel.

        Args:
            hwm: TargetPlatformModel to set as the current TargetPlatformModel to access and use.

        """
        self.hwm = hwm


# Use a single instance for the current model.
_current_hardware_model = CurrentTPModel()
