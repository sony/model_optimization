# Copyright 2022 Sony Semiconductors Israel, Inc. All rights reserved.
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
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.python.util.object_identity import Reference as TFReference
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit.default_8bit_quantize_configs import \
    NoOpQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.constants import THRESHOLD, RANGE_MIN, RANGE_MAX, SIGNED
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.target_platform import QuantizationMethod
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder, \
    is_layer_fake_quant, get_node_name_from_layer
from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
from model_compression_toolkit.core.quantizers.keras.tf_fq_quantizer import TFFakeQuantQuantizer
from model_compression_toolkit.core.quantizers.keras.uniform_quantizer import UniformQuantizer
from model_compression_toolkit.core.quantizers.keras.weights_quantize_config import WeightsQuantizeConfig, \
    ActivationQuantizeConfig, WeightsActivationQuantizeConfig


class FullyQuantizedKerasModelBuilder(KerasModelBuilder):
    """
    Builder of fully-quantized Keras models.
    """

    def __init__(self,
                 graph: common.Graph,
                 append2output=None,
                 fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                 return_float_outputs: bool = False):
        """

        Args:
            graph: Graph to build the model from.
            append2output: Nodes to append to model's output.
            fw_info: Information about the specific framework of the model that is built.
            return_float_outputs: Whether the model returns float tensors or not.
        """

        super().__init__(graph,
                         append2output,
                         fw_info,
                         return_float_outputs)

    def _quantize_node_activations(self,
                                   node: BaseNode,
                                   input_tensors: List[TFReference]) -> List[TFReference]:
        """
        Quantize node's activation given input tensors.

        Args:
            node: Node to quantize its outputs.
            input_tensors: Input tensors of the node.

        Returns:
            Output of the node.

        """
        return input_tensors
        # return node.final_activation_quantization_cfg.quantize_node_output(input_tensors)

    def build_model(self) -> Tuple[Model, UserInformation]:
        """
        Build a Keras mixed-precision model and return it.
        Returns: Mixed-precision Keras model.

        """
        model, user_info = super().build_model()

        def _wrap_layer_with_quantize_config(layer):

            nodes = self.graph.find_node_by_name(get_node_name_from_layer(layer))

            if len(nodes) == 1:
                node = nodes[0]
                return QuantizeWrapper(layer, self._get_quantization_config(node))

            elif is_layer_fake_quant(layer):
                return layer

            else:
                raise Exception(
                    f'Mismatch between keras model and graph cant find node named: '
                    f'{get_node_name_from_layer(layer)}')

        # clone each layer in the model and apply _wrap_layer_with_quantize_config to the layer.
        model = tf.keras.models.clone_model(model,
                                            input_tensors=None,
                                            clone_function=_wrap_layer_with_quantize_config)

        return model, user_info

    def _get_quantization_config(self,
                                 node: BaseNode):
        if node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
            return WeightsQuantizeConfig(weight_attrs=self.fw_info.get_kernel_op_attributes(node.type),
                                         w_quantizer=self._get_weights_quantizer_for_node(node))

        elif not node.is_weights_quantization_enabled() and node.is_activation_quantization_enabled():
            return ActivationQuantizeConfig(activation_quantizer=self._get_activations_quantizer_for_node(node))

        elif not node.is_weights_quantization_enabled() and not node.is_activation_quantization_enabled():
            return NoOpQuantizeConfig()

        return WeightsActivationQuantizeConfig(activation_quantizer=self._get_activations_quantizer_for_node(node),
                                               w_quantizer=self._get_weights_quantizer_for_node(node),
                                               weight_attrs=self.fw_info.get_kernel_op_attributes(node.type))


    def _calculate_delta(self,
                         threshold: np.ndarray,
                         n_bits: int = 8,
                         signed: bool = False) -> np.ndarray:
        """
        Compute the step size of quantized values given the threshold, number of bits
        and whether its signed or unsigned.

        Args:
            threshold: Threshold to compute the step size according to.
            n_bits: Number of bits to compute the step size according to.
            signed: Whether quantization range is signed or not.

        Returns:
            Step size of quantized values according to a threshold, signedness and number of bits.
        """

        return threshold / (2 ** (n_bits - int(signed)))

    def _get_weights_quantizer_for_node(self, node: BaseNode):

        assert node.final_weights_quantization_cfg is not None, f'Can not set quantizer for a node with no final ' \
                                                                f'weights quantization configuration'

        supported_quantizers = [QuantizationMethod.POWER_OF_TWO,
                                QuantizationMethod.SYMMETRIC,
                                QuantizationMethod.UNIFORM]

        node_w_qc = node.final_weights_quantization_cfg
        weights_quantization_method = node_w_qc.weights_quantization_method
        assert weights_quantization_method in supported_quantizers, \
            f'Fully quantized models are now supported for {supported_quantizers} quantization methods, but node ' \
            f'has {weights_quantization_method} quantization method'

        if weights_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
            # TODO: Add assertion for POT case that thresholds are POT
            min_range = -node_w_qc.weights_quantization_params.get(THRESHOLD)
            max_range = node_w_qc.weights_quantization_params.get(THRESHOLD) - self._calculate_delta(
                node_w_qc.weights_quantization_params.get(THRESHOLD),
                n_bits=node_w_qc.weights_n_bits,
                signed=True)
        elif weights_quantization_method in [QuantizationMethod.UNIFORM]:
            min_range = node_w_qc.weights_quantization_params.get(RANGE_MIN)
            max_range = node_w_qc.weights_quantization_params.get(RANGE_MAX)
        else:
            raise NotImplemented

        return UniformQuantizer(node_w_qc.weights_n_bits,
                                min_range,
                                max_range,
                                node_w_qc.weights_channels_axis,
                                node_w_qc.weights_per_channel_threshold)


    def _get_activations_quantizer_for_node(self, node: BaseNode):

        assert node.final_activation_quantization_cfg is not None, f'Can not set quantizer for a node with no final ' \
                                                                f'weights quantization configuration'

        supported_quantizers = [QuantizationMethod.POWER_OF_TWO,
                                QuantizationMethod.SYMMETRIC,
                                QuantizationMethod.UNIFORM]

        node_act_qc = node.final_activation_quantization_cfg
        activation_quantization_method = node_act_qc.activation_quantization_method
        assert activation_quantization_method in supported_quantizers, \
            f'Fully quantized models are now supported for {supported_quantizers} quantization methods, but node ' \
            f'has {activation_quantization_method} quantization method'

        if activation_quantization_method in [QuantizationMethod.POWER_OF_TWO, QuantizationMethod.SYMMETRIC]:
            # TODO: Add assertion for POT case that thresholds are POT
            min_range = 0
            if node_act_qc.activation_quantization_params.get(SIGNED):
                min_range = -node_act_qc.activation_quantization_params.get(THRESHOLD)
            max_range = node_act_qc.activation_quantization_params.get(THRESHOLD) - self._calculate_delta(
                node_act_qc.activation_quantization_params.get(THRESHOLD),
                n_bits=node_act_qc.activation_n_bits,
                signed=node_act_qc.activation_quantization_params.get(SIGNED))
        else:
            raise NotImplemented
        # return partial(tf.quantization.fake_quant_with_min_max_args,
        #                min=min_range,
        #                max=max_range,
        #                num_bits=node_act_qc.activation_n_bits)
        return TFFakeQuantQuantizer(node_act_qc.activation_n_bits,
                                    min_range,
                                    max_range)

        # return UniformQuantizer(node_act_qc.activation_n_bits,
        #                         min_range,
        #                         max_range)
