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


def get_current_target_platform_capabilities():
    """

    Returns: The current TargetPlatformCapabilities that is being used and accessed.

    """
    return _current_target_platform_capabilities.get()


class _CurrentTargetPlatformCapabilities(object):
    """
    Wrapper of the current TargetPlatformCapabilities object that is being accessed and defined.
    """
    def __init__(self):
        super(_CurrentTargetPlatformCapabilities, self).__init__()
        self.tbc = None

    def get(self):
        """

        Returns: The current TargetPlatformCapabilities that is being defined.

        """
        if self.tbc is None:
            raise Exception('TargetPlatformCapabilities is not initialized.')
        return self.tbc

    def reset(self):
        """

        Reset the current TargetPlatformCapabilities so a new TargetPlatformCapabilities can be wrapped and
        used as the current TargetPlatformCapabilities object.

        """
        self.tbc = None

    def set(self, tbc):
        """
        Set and wrap a TargetPlatformCapabilities as the current TargetPlatformCapabilities.

        Args:
            tbc: TargetPlatformCapabilities to set as the current TargetPlatformCapabilities to access and use.

        """
        self.tbc = tbc


# Use a single instance for the current model.
_current_target_platform_capabilities = _CurrentTargetPlatformCapabilities()
