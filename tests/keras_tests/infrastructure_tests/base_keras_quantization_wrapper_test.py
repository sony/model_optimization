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
from typing import Dict

import tensorflow as tf
from keras.models import clone_model
from tensorflow import TensorShape
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core.common.quantization.node_quantization_config import NodeWeightsQuantizationConfig
from model_compression_toolkit import qunatizers_infrastructure as qi, QuantizationConfig
from model_compression_toolkit.core.common.target_platform import OpQuantizationConfig, QuantizationMethod
from tests.common_tests.base_test import BaseTest

from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder

keras = tf.keras
layers = keras.layers


class ZeroQuantizer(qi.BaseKerasQuantizer):
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
                                tensor_shape: TensorShape,
                                name: str,
                                layer: QuantizeWrapper) -> Dict[str, tf.Variable]:
        return

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
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


class BaseKerasInfrastructureTest(BaseTest):
    def __init__(self,
                 unit_test,
                 num_calibration_iter=1,
                 val_batch_size=1,
                 num_of_inputs=1,
                 input_shape=(8, 8, 3)):
        super().__init__(unit_test=unit_test,
                         val_batch_size=val_batch_size,
                         num_calibration_iter=num_calibration_iter,
                         num_of_inputs=num_of_inputs,
                         input_shape=input_shape)

    def get_dispatcher(self, weight_quantizers, activation_quantizers):
        return qi.KerasNodeQuantizationDispatcher(weight_quantizers, activation_quantizers)

    def get_wrapper(self, weight_quantizers, activation_quantizers):
        return qi.KerasQuantizationWrapper(self.create_networks(), self.get_dispatcher(weight_quantizers, activation_quantizers))

    def run_test(self):
        model = self.create_networks()
        conv_layer = model.layers[1]
        nqd = qi.KerasNodeQuantizationDispatcher()
        nqd.add_weight_quantizer('kernel', ZeroQuantizer(weight_quantization_config))
        wrapper = qi.KerasQuantizationWrapper(conv_layer, nqd)
        # init
        # get config
        # from config
        # build
        wrapper.build(self.get_input_shapes())
        (name, weight, quantizer) = wrapper._weight_vars[0]
        self.unit_test.assertTrue(isinstance(wrapper, qi.KerasQuantizationWrapper))
        self.unit_test.assertTrue(isinstance(wrapper.layer, layers.Conv2D))
        self.unit_test.assertTrue(name == 'kernel')
        # self.assertTrue((weight == getattr(wrapper.layer, 'weight')).any())
        # self.assertTrue(isinstance(quantizer, ZeroWeightsQuantizer))
        # y = wrapper(torch.Tensor(np.random.random((4, 3, 224, 224)))) # apply the wrapper on some random inputs
        # self.assertTrue((0 == getattr(wrapper.layer, 'weight')).any()) # check the weight are now quantized
        # self.assertTrue((y[0,:,0,0] == getattr(wrapper.layer, 'bias')).any()) # check the wrapper's outputs are equal to biases

        # call
        # set quantized weights

        a = wrapper(model.layers[1])
        # for layer in model.layers:
        #     wrapper(layer)

        def _wrap(layer):
            _nodes = self.graph.find_node_by_name(layer.name)
            if len(_nodes) == 1:
                return self.wrapper(_nodes[0], layer)

        model = clone_model(self.create_networks(), clone_function=_wrap)
        # model = clone_model(model, clone_function=_wrap)
        # (name, weight, quantizer) = wrapper._weight_vars[0]
        self.assertTrue(isinstance(wrapper, qi.KerasQuantizationWrapper))
        self.assertTrue(isinstance(wrapper.layer, self.create_networks()))
        self.assertTrue(name == 'weight')
        self.assertTrue((weight == getattr(wrapper.layer, 'weight')).any())
        self.assertTrue(isinstance(quantizer, ZeroQuantizer))


class LayerKerasInfrastructureTest(BaseKerasInfrastructureTest):
    def __init__(self, unit_test):
        super().__init__(unit_test)

    def create_networks(self):
        inputs = layers.Input(shape=self.get_input_shapes()[0][1:])
        x = layers.Conv2D(6, 7, use_bias=False)(inputs)
        return keras.Model(inputs=inputs, outputs=x)

