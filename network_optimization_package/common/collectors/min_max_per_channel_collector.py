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

import numpy as np

from network_optimization_package.common.collectors.base_collector import BaseCollector


class MinMaxPerChannelCollector(BaseCollector):
    """
    Class to collect observed mix/max values of tensors that goes through it (passed to update).
    """

    def __init__(self,
                 init_min_value: float = None,
                 init_max_value: float = None,
                 axis=-1):
        """
        Instantiate a collector for collecting min/max values of tensor per-channel.
        Args:
            axis: Compute the min/max values with regard to this axis.
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

        axis = (len(x.shape) - 1) if self.axis == -1 else self.axis  # convert
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
