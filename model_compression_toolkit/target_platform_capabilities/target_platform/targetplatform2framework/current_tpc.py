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


def get_current_tpc():
    """

    Returns: The current TargetPlatformCapabilities that is being used and accessed.

    """
    return _current_tpc.get()


class _CurrentTPC(object):
    """
    Wrapper of the current TargetPlatformCapabilities object that is being accessed and defined.
    """
    def __init__(self):
        super(_CurrentTPC, self).__init__()
        self.tpc = None

    def get(self):
        """

        Returns: The current TargetPlatformCapabilities that is being defined.

        """
        if self.tpc is None:
            raise Exception('TargetPlatformCapabilities is not initialized.')
        return self.tpc

    def reset(self):
        """

        Reset the current TargetPlatformCapabilities so a new TargetPlatformCapabilities can be wrapped and
        used as the current TargetPlatformCapabilities object.

        """
        self.tpc = None

    def set(self, target_platform_capabilities):
        """
        Set and wrap a TargetPlatformCapabilities as the current TargetPlatformCapabilities.

        Args:
            target_platform_capabilities: TargetPlatformCapabilities to set as the current TargetPlatformCapabilities
            to access and use.

        """
        self.tpc = target_platform_capabilities


# Use a single instance for the current model.
_current_tpc = _CurrentTPC()
