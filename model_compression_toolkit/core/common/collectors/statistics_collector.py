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


import math
from copy import deepcopy
from typing import Any, Tuple

import numpy as np

from model_compression_toolkit.core.common.framework_info import FrameworkInfo, ChannelAxis
from model_compression_toolkit.core.common.collectors.histogram_collector import HistogramCollector
from model_compression_toolkit.core.common.collectors.mean_collector import MeanCollector
from model_compression_toolkit.core.common.collectors.min_max_per_channel_collector import MinMaxPerChannelCollector


class BaseStatsCollector(object):
    """
    Base class for statistics collection (contains multiple collectors such as mean collector,
    histogram collector, etc.).
    """

    def require_collection(self) -> bool:
        """
        Returns whether this tensor requires statistics collection or not.
        Should be implemented in extending classes.
        """
        raise Exception(f'require_collection is not implemented in {self.__class__.__name__}')

    def update_statistics(self,
                          x: Any):
        """
        Update statistics in collectors with a new tensor to consider.

        Args:
            x: Tensor.
        """
        raise Exception(f'update_statistics is not implemented in {self.__class__.__name__}')


class StatsCollector(BaseStatsCollector):
    """
    Class to wrap all statistics that are being collected for an input/output node.
    """

    def __init__(self,
                 out_channel_axis: int,
                 init_min_value: float = None,
                 init_max_value: float = None):
        """
        Instantiate three statistics collectors: histogram, mean and min/max per channel.
        Set initial min/max values if are known.

        Args:
            out_channel_axis: Index of output channels.
            init_min_value: Initial min value for min/max stored values.
            init_max_value: Initial max value for min/max stored values.
        """

        super().__init__()
        self.hc = HistogramCollector()
        self.mc = MeanCollector(axis=out_channel_axis)
        self.mpcc = MinMaxPerChannelCollector(init_min_value=init_min_value,
                                              init_max_value=init_max_value,
                                              axis=out_channel_axis)

    def update_statistics(self, x: Any):
        """
        Update statistics in all collectors with a new tensor to consider.

        Args:
            x: Tensor to consider when updating statistics.
        """

        x = standardize_tensor(x)
        self.hc.update(x)
        self.mc.update(x)
        self.mpcc.update(x)

    def get_mean(self) -> np.ndarray:
        """
        Get mean per-channel from mean collector. When its accessed from outside the tensor,
        the scale and shift come into consideration.

        Returns: Mean per-channel from mean collector.
        """

        return self.mc.state

    def get_min_max_values(self) -> Tuple[float, float]:
        """
        Get min/max from collector.
        When its accessed from outside the tensor, the scale and shift come into consideration.

        Returns: Min/max from collector.
        """

        min_value = self.mpcc.min
        max_value = self.mpcc.max

        if not self.mpcc.ignore_init_values:
            if is_number(self.mpcc.init_min_value):
                min_value = self.mpcc.init_min_value

            if is_number(self.mpcc.init_max_value):
                max_value = self.mpcc.init_max_value

        return float(min_value), float(max_value)

    def __repr__(self) -> str:
        """
        Display Tensor with its current and initial mix/max values.
        Returns: String to display.
        """

        return f"<Min:{self.mpcc.min}," \
               f" Max:{self.mpcc.max} " \
               f"with init values " \
               f"Min:{self.mpcc.init_min_value}, " \
               f"Max:{self.mpcc.init_max_value}>"

    def require_collection(self) -> bool:
        """
        Returns: True since tensor requires statistics collection.
        """

        return True


class NoStatsCollector(BaseStatsCollector):
    """
    Class that inherits from base tensor.
    Indicating that for a point in a graph we should not gather statistics.
    """

    def __init__(self):
        super().__init__()

    def update_statistics(self,
                          x: Any):
        """
        Do nothing in BaseTensor method since we are not collecting statistics here.

        Args:
            x: Tensor.
        """

        pass  # pragma: no cover

    def __repr__(self):
        """
        Returns: Display object as "No Quantization".
        """

        return "No Stats Collector"

    def require_collection(self):
        """
        Returns: False since NoTensor does not statistics collection.
        """

        return False


def is_number(num: Any) -> bool:
    """
    Check if a variable is an actual number (and not None or math.inf).
    Args:
        num: Variable to check.

    Returns:
        Whether the vriable is a number or not.
    """
    if num is not None and \
            not math.isinf(num):
        return True
    return False


def standardize_tensor(x: Any) -> np.ndarray:
    """
    Standardize tensors that goes through the collectors before using them.
    Convert them to numpy arrays of float data type.

    Args:
        x: Tensor to standardize.

    Returns:
        Same tensor as numpy ndarray of float data type.
    """
    x = x.astype(np.float)
    if not isinstance(x, np.ndarray) and len(x.shape) == 0:
        x = np.asarray(x)
        x = x.reshape([1])
    return x


def shift_statistics(collector: BaseStatsCollector,
                     shift_value: np.ndarray) -> BaseStatsCollector:
    """
    Shift all statistics in collectors of a statistics collector by a
    value (or a value per-channel).

    Args:
        collector: Statistics collector to shift its collectors.
        shift_value: Value to shift all statistics by.

    Returns:
        New copy of the collector with shifted statistics.

    """

    shifted_collector = deepcopy(collector)
    if isinstance(collector, StatsCollector):
        shifted_collector.mpcc.shift(shift_value)
        shifted_collector.mc.shift(shift_value)
        if shifted_collector.require_collection():
            shifted_collector.hc.shift(shift_value)

    return shifted_collector


def scale_statistics(collector: BaseStatsCollector,
                     scale_value: np.ndarray) -> BaseStatsCollector:
    """
    Scale all statistics in collectors of a statistics collector
    by a factor (or a factor per-channel).

    Args:
        collector: Statistics collector to shift its collectors.
        scale_value: Value to shift all statistics by.

    Returns:
        New copy of the collector with scaled statistics.

    """

    scaled_collector = deepcopy(collector)
    if isinstance(collector, StatsCollector):
        scaled_collector.mpcc.scale(scale_value)
        scaled_collector.mc.scale(scale_value)
        if scaled_collector.require_collection():
            scaled_collector.hc.scale(scale_value)

    return scaled_collector
