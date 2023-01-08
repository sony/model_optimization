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

from typing import Dict

import torch
import torch.nn as nn

from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig, \
    NodeActivationQuantizationConfig
from model_compression_toolkit import qunatizers_infrastructure as qi, QuantizationConfig
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, QuantizationMethod


class ZeroWeightsQuantizer(qi.BasePytorchQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: node quantization config class
        """
        super().__init__(quantization_config,
                         qi.QuantizationTarget.Weights,
                         [qi.QuantizationMethod.POWER_OF_TWO, qi.QuantizationMethod.SYMMETRIC])

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: nn.Module) -> Dict[str, nn.Parameter]:
        return

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool) -> nn.Parameter:

        return inputs * 0


class ZeroActivationsQuantizer(qi.BasePytorchQuantizer):
    """
    Trainable constrained quantizer to quantize a layer inputs.
    """

    def __init__(self, quantization_config: NodeWeightsQuantizationConfig):
        """
        Initialize a TrainableWeightQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: node quantization config class
        """
        super().__init__(quantization_config,
                         qi.QuantizationTarget.Activation,
                         [qi.QuantizationMethod.POWER_OF_TWO, qi.QuantizationMethod.SYMMETRIC])

    def initialize_quantization(self,
                                tensor_shape: torch.Size,
                                name: str,
                                layer: nn.Module) -> Dict[str, nn.Parameter]:
        return

    def __call__(self,
                 inputs: nn.Parameter,
                 training: bool = True) -> nn.Parameter:

        return inputs * 0


def dummy_fn():
    return


op_cfg = OpQuantizationConfig(QuantizationMethod.POWER_OF_TWO,
                              QuantizationMethod.POWER_OF_TWO,
                              8,
                              8,
                              True,
                              True,
                              True,
                              True,
                              1,
                              0,
                              32)
qc = QuantizationConfig()
weight_quantization_config = NodeWeightsQuantizationConfig(qc, op_cfg, dummy_fn, dummy_fn, -1)
activations_quantization_config = NodeActivationQuantizationConfig(qc, op_cfg, dummy_fn, dummy_fn)

op_cfg_uniform = OpQuantizationConfig(QuantizationMethod.UNIFORM,
                                        QuantizationMethod.UNIFORM,
                                        8,
                                        8,
                                        True,
                                        True,
                                        True,
                                        True,
                                        1,
                                        0,
                                        32)
weight_quantization_config_uniform = NodeWeightsQuantizationConfig(qc, op_cfg_uniform, dummy_fn, dummy_fn, -1)
activations_quantization_config_uniform = NodeActivationQuantizationConfig(qc, op_cfg_uniform, dummy_fn, dummy_fn)


class TestPytorchNodeQuantizationDispatcher(unittest.TestCase):

    def test_pytorch_base_quantizer(self):

        with self.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(weight_quantization_config_uniform)
        self.assertEqual(f'Quantization method mismatch expected:[<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM', str(e.exception))

        with self.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(activations_quantization_config_uniform)
        self.assertEqual(f'Quantization method mismatch expected:[<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 3>] and got  QuantizationMethod.UNIFORM', str(e.exception))

        with self.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(activations_quantization_config_uniform)
        self.assertEqual(f'Expect weight quantization got activation', str(e.exception))

        with self.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(weight_quantization_config_uniform)
        self.assertEqual(f'Expect activation quantization got weight', str(e.exception))

        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.assertTrue(quantizer.quantization_config == weight_quantization_config)

        quantizer = ZeroActivationsQuantizer(activations_quantization_config)
        self.assertTrue(quantizer.quantization_config == activations_quantization_config)