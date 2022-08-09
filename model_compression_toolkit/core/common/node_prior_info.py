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

from model_compression_toolkit.core.common.collectors.statistics_collector import is_number


class NodePriorInfo:
    """
    Class to wrap all prior information we have on a node.
    """

    def __init__(self,
                 min_output: float = None,
                 max_output: float = None,
                 mean_output: float = None,
                 std_output: float = None):
        """
        Initialize a NodePriorInfo object.

        Args:
            min_output: Minimal output value of the node.
            max_output: Maximal output value of the node.
            mean_output: Mean output value of the node.
            std_output: Std output value of the node.
        """

        self.min_output = min_output
        self.max_output = max_output
        self.mean_output = mean_output
        self.std_output = std_output

    def is_output_bounded(self) -> bool:
        """

        Returns: Whether the node's output is bounded within a known range or not.

        """
        return is_number(self.min_output) and is_number(self.max_output)
