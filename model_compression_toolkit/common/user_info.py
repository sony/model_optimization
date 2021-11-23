# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import List


class UserInformation(object):
    """
    Class to wrap information the user gets after passing the model
    through the model optimization process. For example, what is the scale factor
    the input data should get scaled by before passing it to the model (when enabling
    input scaling during the process).
    """

    def __init__(self):
        self.input_scale = 1
        self.gptq_info_dict = dict()
        self.mixed_precision_cfg = None

    def set_input_scale(self, scale_value: float):
        """
        Set the UserInformation an input scale value.

        Args:
            scale_value: Scale factor to store in the UserInformation.

        """
        self.input_scale = scale_value

    def set_mixed_precision_cfg(self, mp_cfg:List[int]):
        self.mixed_precision_cfg = mp_cfg

