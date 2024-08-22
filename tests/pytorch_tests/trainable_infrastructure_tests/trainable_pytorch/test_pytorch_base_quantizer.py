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
from model_compression_toolkit.trainable_infrastructure.common.trainable_quantizer_config import \
    TrainableQuantizerWeightsConfig, TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.pytorch.base_pytorch_quantizer import \
    BasePytorchTrainableQuantizer
from tests.pytorch_tests.trainable_infrastructure_tests.base_pytorch_trainable_infra_test import \
    BasePytorchInfrastructureTest, ZeroWeightsQuantizer, ZeroActivationsQuantizer
from model_compression_toolkit.target_platform_capabilities.target_platform import QuantizationMethod


class TestPytorchBaseWeightsQuantizer(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_weights_quantization_config(self):
        return TrainableQuantizerWeightsConfig(weights_quantization_method=QuantizationMethod.UNIFORM,
                                               weights_n_bits=8,
                                               weights_quantization_params={},
                                               enable_weights_quantization=True,
                                               weights_channels_axis=0,
                                               weights_per_channel_threshold=True,
                                               min_threshold=0)

    def run_test(self):

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch. Expected methods: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 2>], received: QuantizationMethod.UNIFORM.', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroWeightsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Expected weight quantization configuration; received activation quantization instead.', str(e.exception))

        weight_quantization_config = super(TestPytorchBaseWeightsQuantizer, self).get_weights_quantization_config()
        quantizer = ZeroWeightsQuantizer(weight_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == weight_quantization_config)


class TestPytorchBaseActivationQuantizer(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def get_activation_quantization_config(self):
        return TrainableQuantizerActivationConfig(activation_quantization_method=QuantizationMethod.UNIFORM,
                                                  activation_n_bits=8,
                                                  activation_quantization_params={},
                                                  enable_activation_quantization=True,
                                                  min_threshold=0)

    def run_test(self):

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_activation_quantization_config())
        self.unit_test.assertEqual(f'Quantization method mismatch. Expected methods: [<QuantizationMethod.POWER_OF_TWO: 0>, <QuantizationMethod.SYMMETRIC: 2>], received: QuantizationMethod.UNIFORM.', str(e.exception))

        with self.unit_test.assertRaises(Exception) as e:
            ZeroActivationsQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(f'Expected activation quantization configuration; received weight quantization instead.', str(e.exception))

        activation_quantization_config = super(TestPytorchBaseActivationQuantizer, self).get_activation_quantization_config()
        quantizer = ZeroActivationsQuantizer(activation_quantization_config)
        self.unit_test.assertTrue(quantizer.quantization_config == activation_quantization_config)


class _TestQuantizer(BasePytorchTrainableQuantizer):
    def __init__(self, quantization_config: TrainableQuantizerWeightsConfig):
        super().__init__(quantization_config)


class TestPytorchQuantizerWithoutMarkDecorator(BasePytorchInfrastructureTest):

    def __init__(self, unit_test):
        super().__init__(unit_test)

    def run_test(self):
        # create instance of dummy _TestQuantizer. Should throw exception because it is not marked with @mark_quantizer.
        with self.unit_test.assertRaises(Exception) as e:
            test_quantizer = _TestQuantizer(self.get_weights_quantization_config())
        self.unit_test.assertEqual(
            "Quantizer class inheriting from 'BaseTrainableQuantizer' is improperly defined. "
            "Ensure it includes the '@mark_quantizer' decorator and is correctly applied.",
            str(e.exception))
