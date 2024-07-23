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
from functools import partial
from typing import Callable, Dict

import numpy as np

from model_compression_toolkit.data_generation.common.enums import SchedulerType
from model_compression_toolkit.data_generation.keras.optimization_functions.lr_scheduler import \
    ReduceLROnPlateau


def get_reduce_lr_on_plateau_scheduler(n_iter: int, initial_lr: float):
    """
    Create a custom ReduceLROnPlateau learning rate scheduler.

    Args:
        n_iter (int): Total number of iterations.
        initial_lr (float): Initial learning rate.

    Returns:
        callable: Partial function for creating CustomReduceLROnPlateau scheduler.
    """
    return partial(ReduceLROnPlateau, min_lr=1e-4, factor=0.5, patience=int(n_iter / 50))



# Define a dictionary that maps scheduler types to functions for creating schedulers.
scheduler_step_function_dict: Dict[SchedulerType, Callable] = {
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_lr_on_plateau_scheduler,  # ReduceLROnPlateau scheduler.
}
