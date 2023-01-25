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

from model_compression_toolkit.core.common.logger import Logger

def get_current_tp_model():
    """

    Returns: The current TargetPlatformModel that is being used and accessed.

    """
    return _current_tp_model.get()


class CurrentTPModel:
    """
    Wrapper of the current TargetPlatformModel object that is being accessed and defined.
    """

    def __init__(self):
        super(CurrentTPModel, self).__init__()
        self.tp_model = None

    def get(self):
        """

        Returns: The current TargetPlatformModel that is being defined.

        """
        if self.tp_model is None:
            Logger.error('Target platform model is not initialized.')  # pragma: no cover
        return self.tp_model

    def reset(self):
        """

        Reset the current TargetPlatformModel so a new TargetPlatformModel can be wrapped and
        used as the current TargetPlatformModel object.

        """
        self.tp_model = None

    def set(self, tp_model):
        """
        Set and wrap a TargetPlatformModel as the current TargetPlatformModel.

        Args:
            tp_model: TargetPlatformModel to set as the current TargetPlatformModel to access and use.

        """
        self.tp_model = tp_model


# Use a single instance for the current model.
_current_tp_model = CurrentTPModel()
