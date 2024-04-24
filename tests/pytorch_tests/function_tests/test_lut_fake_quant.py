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
import unittest

import numpy as np
import torch

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from model_compression_toolkit.core.pytorch.quantizer.lut_fake_quant import activation_lut_kmean_quantizer

from model_compression_toolkit.constants import SIGNED, LUT_VALUES, THRESHOLD

class TestPytorchActivationLutQuantizer(unittest.TestCase):

    def test_lut_pot_signed_quantizer(self):
        lut_values = [-25, 25]
        thresholds = [4.]
        num_bits = 3
        signed = True
        lut_values_bitwidth = 8

        quantizer = activation_lut_kmean_quantizer(activation_n_bits=num_bits,
                                                   quantization_params={SIGNED: signed,
                                                                        LUT_VALUES: np.asarray(lut_values),
                                                                        THRESHOLD: thresholds[0]})

        # Initialize a random input to quantize between -50 to 50.
        input_tensor = torch.rand(1, 3, 3, 3) * 100 - 50
        fake_quantized_tensor = quantizer(input_tensor.to(get_working_device()))

        # Using a signed quantization, so we expect all values to be between -abs(max(threshold))
        # and abs(max(threshold))
        max_threshold = np.max(np.abs(thresholds))
        delta_threshold = 1 / (2 ** (lut_values_bitwidth - int(signed)))

        fake_quantized_tensor = fake_quantized_tensor.detach().cpu().numpy()

        self.assertTrue(np.max(fake_quantized_tensor) <= (max_threshold - delta_threshold), f'Quantized values should not contain values greater than maximal threshold ')
        self.assertTrue(np.min(fake_quantized_tensor) >= -max_threshold, f'Quantized values should not contain values lower than minimal threshold ')
        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits, f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has {len(np.unique(fake_quantized_tensor))} unique values')

        quant_tensor_values = np.asarray(lut_values) / (2 ** (lut_values_bitwidth - int(signed))) * np.asarray(thresholds)
        self.assertTrue(len(np.unique(fake_quantized_tensor)) <= 2 ** num_bits, f'Quantized tensor expected to have no more than {2 ** num_bits} unique values but has {len(np.unique(fake_quantized_tensor))} unique values')
        self.assertTrue(np.all(np.unique(fake_quantized_tensor) == np.sort(quant_tensor_values)))

        # Check quantized tensor assigned correctly
        clip_max = 2 ** (lut_values_bitwidth - 1) - 1
        clip_min = -2 ** (lut_values_bitwidth - 1)

        tensor = torch.clip((input_tensor / np.asarray(thresholds)) * (2 ** (lut_values_bitwidth - int(signed))), min=clip_min, max=clip_max)
        tensor = tensor.unsqueeze(-1)
        expanded_lut_values = np.asarray(lut_values).reshape([*[1 for _ in range(len(tensor.shape) - 1)], -1])
        lut_values_assignments = torch.argmin(torch.abs(tensor - expanded_lut_values), dim=-1)
        centers = np.asarray(lut_values).flatten()[lut_values_assignments]
        self.assertTrue(np.all(centers / (2 ** (lut_values_bitwidth - int(signed))) * thresholds == fake_quantized_tensor), "Quantized tensor values weren't assigned correctly")

        # Assert some values are negative (signed quantization)
        self.assertTrue(np.any(fake_quantized_tensor < 0), f'Expected some values to be negative but quantized tensor is {fake_quantized_tensor}')
