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
import torch
import pytest

from model_compression_toolkit.trainable_infrastructure.pytorch.annealing_schedulers import LinearAnnealingScheduler


def test_linear_annealing():
    scheduler = LinearAnnealingScheduler(t_start=10, t_end=35, initial_val=3.4, target_val=-1.6)
    for t in [0, 9, 10]:
        assert _isclose(scheduler(t), 3.4)

    for t in [35, 36, 1000]:
        assert _isclose(scheduler(t), -1.6)

    assert _isclose(scheduler(11), 3.2)
    assert _isclose(scheduler(27), 0.)
    assert _isclose(scheduler(34), -1.4)


def test_linear_annealing_ascending():
    scheduler = LinearAnnealingScheduler(t_start=0, t_end=5, initial_val=-0.5, target_val=1.5)
    assert _isclose(scheduler(0), -0.5)
    assert _isclose(scheduler(1), -0.1)
    assert _isclose(scheduler(4), 1.1)
    assert _isclose(scheduler(5), 1.5)


@pytest.mark.parametrize('start', [5, -1])
def test_invalid(start):
    with pytest.raises(ValueError):
        LinearAnnealingScheduler(t_start=start, t_end=4, initial_val=1, target_val=0)


def _isclose(x, y):
    return torch.isclose(x, torch.tensor(y))
