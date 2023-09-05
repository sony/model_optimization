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
from typing import Callable, Any, Dict, Tuple

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from model_compression_toolkit.data_generation.common.enums import SchedulerType


def get_reduce_lr_on_plateau_scheduler(n_iter: int) -> Callable:
    """
    Get a ReduceLROnPlateau scheduler.

    Args:
        n_iter (int): The number of iterations.

    Returns:
        Callable: A partial function to create ReduceLROnPlateau scheduler with specified parameters.
    """
    return partial(ReduceLROnPlateau, min_lr=1e-4, factor=0.5, patience=int(n_iter / 50))

def get_step_lr_scheduler(n_iter: int) -> Callable:
    """
    Get a StepLR scheduler.

    Args:
        n_iter (int): The number of iterations.

    Returns:
        Callable: A partial function to create StepLR scheduler with specified parameters.
    """
    return partial(StepLR, step_size=int(n_iter / 3))

def reduce_lr_on_platu_step_fn(scheduler: Any, i_iter: int, loss_value: float):
    """
    Step function for the ReduceLROnPlateau type scheduler.

    Args:
        scheduler (Any): The scheduler.
        i_iter (int): The current iteration.
        loss_value (float): The current loss value.
    """
    scheduler.step(loss_value)

def scheduler_step_fn(scheduler: Any, i_iter: int, loss_value: float):
    """
    Step function for the StepLR scheduler.

    Args:
        scheduler (Any): The scheduler.
        i_iter (int): The current iteration.
        loss_value (float): The current loss value.
    """
    scheduler.step()


# Dictionary of scheduler functions and their corresponding step functions
scheduler_step_function_dict: Dict[SchedulerType, Tuple[Callable, Callable]] = {
    SchedulerType.REDUCE_ON_PLATEAU: (get_reduce_lr_on_plateau_scheduler, reduce_lr_on_platu_step_fn),
    SchedulerType.STEP: (get_step_lr_scheduler, scheduler_step_fn),
}