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

import numpy as np

from model_compression_toolkit.common.collectors.base_collector import BaseCollector
from model_compression_toolkit.common.framework_info import ChannelAxis


class MinMaxPerChannelCollector(BaseCollector):
    """
    Class to collect observed mix/max values of tensors that goes through it (passed to update).
    """

    def __init__(self,
                 axis: ChannelAxis,
                 init_min_value: float = None,
                 init_max_value: float = None):
        """
        Instantiate a collector for collecting min/max values of tensor per-channel.
        Args:
            axis: Compute the min/max values with regard to this axis.
            init_max_value: Initial maximal output value.
            init_min_value: Initial minimal output value.
        """
        super().__init__()
        self.axis = axis
        self.state = None
        self.init_min_value = init_min_value
        self.init_max_value = init_max_value
        self.ignore_init_values = False

    def scale(self, scale_factor: np.ndarray):
        """
        Scale all statistics in collector by some value.
        Since min/max are collected per-channel, they can be scaled either by a single factor or a
        scaling factor per-channel.

        Args:
            scale_factor: Factor to scale all collector's statistics by.

        """

        scale_per_channel = scale_factor.flatten().shape[0] > 1  # current scaling is per channel or not
        self.state = np.transpose(np.transpose(self.state) * scale_factor)

        if scale_per_channel:
            self.ignore_init_values = True
        else:
            if self.init_max_value is not None:
                self.init_max_value *= scale_factor
            if self.init_min_value is not None:
                self.init_min_value *= scale_factor

    def shift(self, shift_value: np.ndarray):
        """
        Shift all statistics in collector by some value.
        Since min/max are collected per-channel, they can be shifted either by a single value or a
        shifting value per-channel.

        Args:
            shift_value: Value to shift all collector's statistics by.
        """

        shift_per_channel = shift_value.flatten().shape[0] > 1  # current shifting is per channel or not
        self.state = np.transpose(np.transpose(self.state) + shift_value)

        if shift_per_channel:
            self.ignore_init_values = True
        else:
            if self.init_max_value is not None:
                self.init_max_value += shift_value
            if self.init_min_value is not None:
                self.init_min_value += shift_value

    @property
    def min(self) -> float:
        """
        Returns: Minimal value the collector observed (in general, not per-channel).
        """

        self.validate_data_correctness()
        return None if self.state is None else np.min(self.state[:, 1])

    @property
    def max(self) -> float:
        """
        Returns: Maximal value the collector observed (in general, not per-channel).
        """

        self.validate_data_correctness()
        return None if self.state is None else np.max(self.state[:, 0])

    @property
    def max_per_channel(self) -> np.ndarray:
        """
        Returns: Maximal value the collector observed per-channel.
        """

        self.validate_data_correctness()
        return self.state[:, 0]

    @property
    def min_per_channel(self) -> np.ndarray:
        """
        Returns: Minimal value the collector observed per-channel.
        """

        self.validate_data_correctness()
        return self.state[:, 1]

    def update(self,
               x: np.ndarray):
        """
        Update the min/max values the collector holds using a new tensor x to consider.

        Args:
            x: Tensor that goes through the collector and needs to be considered in the min/max computation.
        """

        axis = (len(x.shape) - 1) if self.axis == ChannelAxis.NHWC else self.axis.NCHW.value  # convert
        n = x.shape[axis]
        transpose_index = [axis, *[i for i in range(len(x.shape)) if i != axis]]
        x_reshape = np.reshape(np.transpose(x, transpose_index), [n, -1])
        if self.state is None:
            x_max = np.max(x_reshape, axis=-1)
            x_min = np.min(x_reshape, axis=-1)
        else:
            x_max = np.maximum(np.max(x_reshape, axis=-1), self.state[:, 0])
            x_min = np.minimum(np.min(x_reshape, axis=-1), self.state[:, 1])
        self.state = np.stack([x_max, x_min], axis=-1)
