# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import unittest

import torch
from mct_quantizers.pytorch.quantizers import ActivationSymmetricInferableQuantizer, ActivationUniformInferableQuantizer

from model_compression_toolkit.gptq.pytorch.quantizer.activation.ste_activation import \
    STEActivationSymmetricGPTQTrainableQuantizer, STEActivationUniformGPTQTrainableQuantizer


class ActivationQuantTest(unittest.TestCase):
    def test_symmetric(self):
        kwargs = dict(num_bits=3, threshold=[1.5], signed=False)
        q_inferable = ActivationSymmetricInferableQuantizer(**kwargs)
        q = STEActivationSymmetricGPTQTrainableQuantizer(**kwargs)
        self._run_test(q_inferable, q)

    def test_uniform(self):
        kwargs = dict(num_bits=5, min_range=[-3], max_range=[1.5])
        q_inferable = ActivationUniformInferableQuantizer(**kwargs)
        q = STEActivationUniformGPTQTrainableQuantizer(**kwargs)
        self._run_test(q_inferable, q)

    def _run_test(self, q_infer, q):
        x = torch.randn((5, 10, 20), requires_grad=True, generator=torch.Generator().manual_seed(42))
        y_infer = q_infer(x)
        y = q(x)
        self.assertTrue(torch.equal(y_infer, y))
        # if autograd cannot propagate, will crash
        torch.autograd.grad(y.mean(), x)
