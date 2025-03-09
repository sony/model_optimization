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
import pytest

from model_compression_toolkit.gptq import QFractionLinearAnnealingConfig


def test_linear_annealing_cfg_validation():
    with pytest.raises(ValueError, match='Expected.* target_q_fraction <= 1'):
        QFractionLinearAnnealingConfig(initial_q_fraction=0.1, target_q_fraction=1.1, start_step=0, end_step=None)

    with pytest.raises(ValueError, match='Expected.* 0 <= initial_q_fraction'):
        QFractionLinearAnnealingConfig(initial_q_fraction=-0.1, target_q_fraction=-0.9, start_step=0, end_step=100)

    with pytest.raises(ValueError, match='Expected.* initial_q_fraction < target_q_fraction'):
        QFractionLinearAnnealingConfig(initial_q_fraction=0.1, target_q_fraction=0.1, start_step=0, end_step=100)

    with pytest.raises(ValueError, match='Expected.* initial_q_fraction < target_q_fraction'):
        QFractionLinearAnnealingConfig(initial_q_fraction=0.2, target_q_fraction=0.1, start_step=0, end_step=100)

    with pytest.raises(ValueError, match='Expected.* start_step >= 0'):
        QFractionLinearAnnealingConfig(initial_q_fraction=0, target_q_fraction=1, start_step=-1, end_step=100)

    with pytest.raises(ValueError, match='Expected.* start_step < end_step'):
        QFractionLinearAnnealingConfig(initial_q_fraction=0, target_q_fraction=1, start_step=100, end_step=100)

    with pytest.raises(ValueError, match='Expected.* start_step < end_step'):
        QFractionLinearAnnealingConfig(initial_q_fraction=0, target_q_fraction=1, start_step=100, end_step=99)
