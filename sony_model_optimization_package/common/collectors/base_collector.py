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


class BaseCollector(object):
    """
    Base class for statistics collection object.
    """

    def __init__(self):
        # When manipulation statistics in a granularity they were not collected by, the data is invalid.
        self.is_legal = True

    def scale(self, scale_factor: np.ndarray):
        """
        Scale all statistics in collector by some factor.
        Args:
            scale_factor: Factor to scale all collector's statistics by.

        """

        raise Exception(f'{self.__class__.__name__} needs to implement scale operation for its state.')

    def shift(self, shift_value: np.ndarray):
        """
        Shift all statistics in collector by some value.
        Args:
            shift_value: Value to shift all collector's statistics by.

        """

        raise Exception(f'{self.__class__.__name__} needs to implement shift operation for its state.')

    def update_legal_status(self, is_illegal: bool):
        """
        If statistics were manipulated in a granularity they were not collected by, the data is invalid,
        and its legal status should be tracked after each manipulation.
        Args:
            is_illegal: Whether current info is invalid or not.

        """

        self.is_legal = self.is_legal and not is_illegal

    def validate_data_correctness(self):
        """
        Verify the collector's statistics were manipulated in a granularity they were collected by.
        If the statistics are invalid, an exception is raised.
        """

        if not self.is_legal:
            raise Exception(f'{self.__class__.__name__} was manipulated per-channel,'
                            'but collected per-tensor. Data is invalid.')
