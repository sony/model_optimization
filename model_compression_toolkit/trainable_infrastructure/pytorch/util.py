# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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
from functools import cache
from typing import Callable

from tqdm import tqdm


@cache
def get_total_grad_steps(representative_data_gen: Callable) -> int:
    # dry run on the representative dataset to count number of batches
    num_batches = 0
    for _ in tqdm(representative_data_gen(), "Estimating representative dataset size"):
        num_batches += 1
    return num_batches


