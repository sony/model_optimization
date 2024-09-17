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
from unittest.mock import Mock

import pytest
import torch

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.trainable_infrastructure.pytorch.annealing_schedulers import LinearAnnealingScheduler
from model_compression_toolkit.gptq import GradientPTQConfig, GradualActivationQuantizationConfig, QFractionLinearAnnealingConfig
from model_compression_toolkit.gptq.pytorch.quantizer.gradual_activation_quantization import (
    GradualActivationQuantizerWrapper, get_gradual_activation_quantizer_wrapper_factory)


@pytest.fixture
def x():
    return torch.randn((2, 5, 6, 7), generator=torch.Generator().manual_seed(42)).to(device=get_working_device())


class Quantizer:
    def __call__(self, x, training):
        self.training = training
        return 3*x + 1


class TestGradualActivationQuantization:

    def test_gradual_act_quant_wrapper(self, x):
        quantizer = Quantizer()
        qw = GradualActivationQuantizerWrapper(quantizer, q_fraction_scheduler=lambda t: t / (t + 1))

        y0, y1, y2 = [qw(x) for _ in range(3)]
        assert torch.equal(y0, x)  # t=0
        assert torch.allclose(y1, 0.5 * x + (1.5 * x + 0.5))  # t=1
        assert torch.allclose(y2, x / 3 + (2 * x + 2 / 3))  # t=2
        assert quantizer.training is True

        _ = qw(x, False)
        assert quantizer.training is False  # correct flag was propagated

    def test_factory_no_qdrop(self):
        quantizer_wrapper, quantizer = self._run_factory_test(qdrop_cfg=None, get_grad_steps_fn=None)
        assert quantizer_wrapper is quantizer

    @pytest.mark.parametrize('end_step', (20, None))
    def test_factory_linear(self, x, end_step):
        qdrop_cfg = GradualActivationQuantizationConfig(
            QFractionLinearAnnealingConfig(initial_q_fraction=0.3, target_q_fraction=0.8, start_step=10, end_step=end_step)
        )

        def get_total_steps():
            if end_step is None:
                return 50
            assert False  # should not be called if end_step is passed

        quantizer_wrapper, quantizer = self._run_factory_test(qdrop_cfg, get_total_steps)

        scheduler = quantizer_wrapper.q_fraction_scheduler
        assert isinstance(scheduler, LinearAnnealingScheduler)
        exp_end_step = 50 if end_step is None else end_step
        assert scheduler.t_start == 10
        assert scheduler.t_end == exp_end_step
        assert scheduler.initial_val == 0.3
        assert scheduler.target_val == 0.8

        y = [quantizer_wrapper(x) for _ in range(exp_end_step+1)]
        assert torch.allclose(y[9], 0.7 * x + 0.3 * quantizer(x, True))
        assert torch.allclose(y[10], 0.7 * x + 0.3 * quantizer(x, True))
        assert torch.allclose(y[-1], 0.2 * x + 0.8 * quantizer(x, True))

    def test_factory_linear_common_case(self, x):
        # validate that we actually implemented the right thing - on first call float input, on last call fully quantized
        qdrop_cfg = GradualActivationQuantizationConfig(
            QFractionLinearAnnealingConfig(initial_q_fraction=0, target_q_fraction=1, start_step=0, end_step=None)
        )
        quantizer_wrapper, quantizer = self._run_factory_test(qdrop_cfg, lambda: 15)
        y0, *_, y_last = [quantizer_wrapper(x) for _ in range(16)]
        assert torch.equal(y0, x)
        assert torch.allclose(y_last, quantizer(x, True))

    def _run_factory_test(self, qdrop_cfg, get_grad_steps_fn):
        # Mocks are used to just pass anything
        gptq_cfg = GradientPTQConfig(n_epochs=5, optimizer=Mock(), loss=Mock(),
                                     gradual_activation_quantization_config=qdrop_cfg)
        factory = get_gradual_activation_quantizer_wrapper_factory(gptq_cfg, get_grad_steps_fn)
        quantizer = Quantizer()
        quantizer_wrapper = factory(quantizer)
        return quantizer_wrapper, quantizer
