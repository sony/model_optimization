# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Any

import numpy as np

from model_compression_toolkit.core.common import Logger


def validate_uniform_min_max_ranges(min_range: Any, max_range: Any) -> None:
    """
    Validate min/max ranges in uniform quantizers are valid

    Args:
        min_range: min range list to check
        max_range: max range list to check

    """
    assert isinstance(min_range, list), f'Expected min_range to be of type list but is {type(min_range)}'
    assert isinstance(max_range, list), f'Expected max_range to be of type list but is {type(max_range)}'

    assert all([isinstance(x, (float, np.float32, np.float64)) for x in
                min_range]), f'Expected min_range list to contain float values but found {[type(x) for x in min_range]}'
    assert all([isinstance(x, (float, np.float32, np.float64)) for x in
                max_range]), f'Expected max_range list to contain float values but found {[type(x) for x in max_range]}'

    assert len(min_range) == len(
        max_range), f'Expected min/max values to have the same length but min shape: {len(min_range)} and max shape: ' \
                    f'{len(max_range)}'

    # Convert min/max to numpy arrays
    min_range, max_range = np.asarray(min_range), np.asarray(max_range)
    assert np.all(max_range > min_range), f'Expected max_range to be bigger than min_range!'


def validate_adjusted_min_max_ranges(min_range: Any,
                                     max_range: Any,
                                     adj_min:Any,
                                     adj_max:Any) -> None:
    """
    Validate adjusted min/max ranges in uniform quantization are valid

    Args:
        min_range: original min range
        max_range: original max range
        adj_min: adjusted min range
        adj_max: adjusted max range

    """

    assert np.all(adj_min <= 0) and np.all(
        adj_max >= 0), f'Expected zero to be in the range, got min_range={adj_min}, max_range={adj_max}'
    if not np.isclose(np.linalg.norm(adj_min - min_range), 0, atol=1e-6) or not np.isclose(np.linalg.norm(adj_max - max_range), 0, atol=1e-6):
        Logger.warning(f"Adjusting (min_range, max_range) from ({min_range},{max_range}) to ({adj_min},{adj_max})")  # pragma: no cover