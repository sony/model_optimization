# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import math
from copy import deepcopy
from typing import Any, Tuple

import numpy as np

from network_optimization_package.common.collectors.histogram_collector import HistogramCollector
from network_optimization_package.common.collectors.mean_collector import MeanCollector
from network_optimization_package.common.collectors.min_max_per_channel_collector import MinMaxPerChannelCollector


class BaseStatsContainer(object):
    """
    Base class for statistics collection container (contain multiple statistics collector such as mean collector,
    histogram collector, etc.).
    """
    def __init__(self):
        # Disable histogram collection. Enable in specific collectors if needed
        self.collect_histogram = False
        self.use_min_max = False

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


class StatsContainer(BaseStatsContainer):
    """
    Class to wrap all statistics that are being collected for an input/output node.
    """

    def __init__(self,
                 init_min_value: float = None,
                 init_max_value: float = None):
        """
        Instantiate three statistics collectors: histogram, mean and min/max per channel.
        Set initial min/max values if are known.

        Args:
            init_min_value: Initial min value for min/max stored values.
            init_max_value: Initial max value for min/max stored values.
        """

        super().__init__()
        self.use_min_max = is_number(init_min_value) and is_number(init_max_value)
        self.collect_histogram = True
        if self.collect_histogram:
            self.hc = HistogramCollector()
        self.mc = MeanCollector()
        self.mpcc = MinMaxPerChannelCollector(init_min_value=init_min_value,
                                              init_max_value=init_max_value)

    def update_statistics(self, x: Any):
        """
        Update statistics in all collectors with a new tensor to consider.

        Args:
            x: Tensor to consider when updating statistics.
        """

        x = standardize_tensor(x)
        if self.collect_histogram:
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


class NoStatsContainer(BaseStatsContainer):
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


def shift_statistics(collector: BaseStatsContainer,
                     shift_value: np.ndarray) -> BaseStatsContainer:
    """
    Shift all statistics in collectors of a statistics container by a
    value (or a value per-channel).

    Args:
        collector: Statistics container to shift its collectors.
        shift_value: Value to shift all statistics by.

    Returns:
        New copy of the container with shifted statistics.

    """

    shifted_collector = deepcopy(collector)
    if isinstance(collector, StatsContainer):
        shifted_collector.mpcc.shift(shift_value)
        shifted_collector.mc.shift(shift_value)
        if shifted_collector.collect_histogram:
            shifted_collector.hc.shift(shift_value)

    return shifted_collector


def scale_statistics(collector: BaseStatsContainer,
                     scale_value: np.ndarray) -> BaseStatsContainer:
    """
    Scale all statistics in collectors of a statistics container
    by a factor (or a factor per-channel).

    Args:
        collector: Statistics container to shift its collectors.
        scale_value: Value to shift all statistics by.

    Returns:
        New copy of the container with scaled statistics.

    """

    scaled_collector = deepcopy(collector)
    if isinstance(collector, StatsContainer):
        scaled_collector.mpcc.scale(scale_value)
        scaled_collector.mc.scale(scale_value)
        if scaled_collector.collect_histogram:
            scaled_collector.hc.scale(scale_value)

    return scaled_collector
