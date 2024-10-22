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
import numpy as np
import tensorflow as tf

from model_compression_toolkit.gptq.common.gradual_activation_quantization import GradualActivationQuantizerWrapper, \
    get_gradual_activation_quantizer_wrapper_factory
from model_compression_toolkit.trainable_infrastructure.keras.annealing_schedulers import KerasLinearAnnealingScheduler
from model_compression_toolkit.gptq import GradientPTQConfig, GradualActivationQuantizationConfig, QFractionLinearAnnealingConfig



@pytest.fixture
def x():
    return tf.random.normal((2, 5, 6, 7), seed=42, dtype=tf.float32)


class Quantizer:
    def __call__(self, x, training):
        self.training = training
        return 3 * x + 1


class TestGradualActivationQuantization:

    def test_gradual_act_quant_wrapper(self, x):
        quantizer = Quantizer()
        qw = GradualActivationQuantizerWrapper(quantizer, q_fraction_scheduler=lambda t: t / (t + 1))

        y0, y1, y2 = [qw(x, training=True) for _ in range(3)]
        assert np.allclose(y0.numpy(), x.numpy())  # t=0
        assert np.allclose(y1.numpy(), 0.5 * x.numpy() + (1.5 * x.numpy() + 0.5))  # t=1
        assert np.allclose(y2.numpy(), x.numpy() / 3 + (2 * x.numpy() + 2 / 3)) # t=2
        assert quantizer.training is True

        _ = qw(x, training=False)
        assert quantizer.training is False  # correct flag was propagated

    def test_factory_no_qdrop(self):
        quantizer_wrapper, quantizer = self._run_factory_test(qdrop_cfg=None, get_grad_steps_fn=None)
        assert quantizer_wrapper is quantizer

    @pytest.mark.parametrize('end_step', (20, None))
    def test_factory_linear(self, x, end_step):
        qdrop_cfg = GradualActivationQuantizationConfig(
            QFractionLinearAnnealingConfig(initial_q_fraction=0.3, target_q_fraction=0.8, start_step=10,
                                           end_step=end_step)
        )

        def get_total_steps():
            if end_step is None:
                return 50
            assert False  # should not be called if end_step is passed

        quantizer_wrapper, quantizer = self._run_factory_test(qdrop_cfg, get_total_steps)

        scheduler = quantizer_wrapper.q_fraction_scheduler
        assert isinstance(scheduler, KerasLinearAnnealingScheduler)
        exp_end_step = 50 if end_step is None else end_step
        assert scheduler.t_start == 10
        assert scheduler.t_end == exp_end_step
        assert scheduler.initial_val == 0.3
        assert scheduler.target_val == 0.8

        y = [quantizer_wrapper(x, training=True) for _ in range(exp_end_step + 1)]

        assert np.allclose(y[9].numpy(), 0.7 * x.numpy() + 0.3 * quantizer(x, training=True).numpy())
        assert np.allclose(y[10].numpy(), 0.7 * x.numpy() + 0.3 * quantizer(x, training=True).numpy())
        assert np.allclose(y[-1].numpy(), 0.2 * x.numpy() + 0.8 * quantizer(x, training=True).numpy())

    def test_factory_linear_common_case(self, x):
        # validate that we actually implemented the right thing - on first call float input, on last call fully quantized
        qdrop_cfg = GradualActivationQuantizationConfig(
            QFractionLinearAnnealingConfig(initial_q_fraction=0, target_q_fraction=1, start_step=0, end_step=None)
        )
        quantizer_wrapper, quantizer = self._run_factory_test(qdrop_cfg, lambda: 15)
        y0, *_, y_last = [quantizer_wrapper(x, training=True) for _ in range(16)]
        assert np.array_equal(y0.numpy(), x.numpy())
        assert np.allclose(y_last.numpy(), quantizer(x, training=True).numpy())

    def _run_factory_test(self, qdrop_cfg, get_grad_steps_fn):
        # Mocks are used to just pass anything
        gptq_cfg = GradientPTQConfig(n_epochs=5, optimizer=Mock(), loss=Mock(),
                                     gradual_activation_quantization_config=qdrop_cfg)
        factory = get_gradual_activation_quantizer_wrapper_factory(gptq_cfg, get_grad_steps_fn, KerasLinearAnnealingScheduler)
        quantizer = Quantizer()
        quantizer_wrapper = factory(quantizer)
        return quantizer_wrapper, quantizer
