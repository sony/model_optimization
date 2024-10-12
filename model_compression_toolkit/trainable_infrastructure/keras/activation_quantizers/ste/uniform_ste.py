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
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from model_compression_toolkit.trainable_infrastructure.common.constants import FQ_MIN, FQ_MAX
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.trainable_infrastructure import TrainingMethod

from mct_quantizers import mark_quantizer, QuantizationMethod, QuantizationTarget
from mct_quantizers.keras.quantizers import BaseKerasInferableQuantizer, ActivationUniformInferableQuantizer

from model_compression_toolkit.qat.keras.quantizer.quant_utils import adjust_range_to_include_zero
from model_compression_toolkit.core.common.quantization.quantizers.quantizers_helpers import fix_range_to_include_zero
from model_compression_toolkit import constants as C

from model_compression_toolkit.trainable_infrastructure import TrainableQuantizerActivationConfig
from model_compression_toolkit.trainable_infrastructure.common.base_trainable_quantizer import VariableGroup
from model_compression_toolkit.trainable_infrastructure.keras.activation_quantizers import BaseKerasActivationTrainableQuantizer


@mark_quantizer(quantization_target=QuantizationTarget.Activation,
                quantization_method=[QuantizationMethod.UNIFORM],
                identifier=TrainingMethod.STE)
class STEUniformActivationTrainableQuantizer(BaseKerasActivationTrainableQuantizer):
    """
    Trainable constrained quantizer to quantize a layer outputs.
    """

    def __init__(self, quantization_config: TrainableQuantizerActivationConfig, freeze_quant_params: bool = False):
        """
        Initialize a STEUniformActivationTrainableQuantizer object with parameters to use
        for the quantization.

        Args:
            quantization_config: trainable quantizer config class
            freeze_quant_params: whether to freeze learnable quantization parameters. This is unused here, since there is not any quantizaiton params that are learned.

        """
        super().__init__(quantization_config, freeze_quant_params)

        self.num_bits = quantization_config.activation_n_bits
        self.min_range = quantization_config.activation_quantization_params[C.RANGE_MIN]
        self.max_range = quantization_config.activation_quantization_params[C.RANGE_MAX]

    def initialize_quantization(self,
                                tensor_shape: TensorShape,
                                name: str,
                                layer: KerasTrainableQuantizationWrapper):
        """
        Add quantizer parameters to the quantizer parameters dictionary

        Args:
            tensor_shape: tensor shape of the quantized tensor.
            name: Tensor name.
            layer: Layer to quantize.
        """
        fq_min = layer.add_weight(
            name + FQ_MIN,
            shape=(),
            initializer=tf.keras.initializers.Constant(-1.0),
            trainable=False)
        fq_min.assign(self.min_range)

        fq_max = layer.add_weight(
            name + FQ_MAX,
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False)
        fq_max.assign(self.max_range)

        # save the quantizer added parameters for later calculations
        self.add_quantizer_variable(FQ_MIN, fq_min, VariableGroup.QPARAMS)
        self.add_quantizer_variable(FQ_MAX, fq_max, VariableGroup.QPARAMS)

    def __call__(self,
                 inputs: tf.Tensor,
                 training: bool):
        """
        Quantize a tensor.
        Args:
            inputs: Input tensor to quantize.
            training: Whether the graph is in training mode.

        Returns:
            The quantized tensor.
        """

        _min = self.get_quantizer_variable(FQ_MIN)
        _max = self.get_quantizer_variable(FQ_MAX)
        _min, _max = adjust_range_to_include_zero(_min, _max, self.num_bits)
        q_tensor = tf.quantization.fake_quant_with_min_max_vars(inputs, _min, _max,
                                                                num_bits=self.num_bits)

        return q_tensor

    def convert2inferable(self) -> BaseKerasInferableQuantizer:
        """
        Convert quantizer to inferable quantizer.

        Returns:
            BaseKerasInferableQuantizer object.
        """
        min_range, max_range = fix_range_to_include_zero(self.get_quantizer_variable(FQ_MIN).numpy(),
                                                         self.get_quantizer_variable(FQ_MAX).numpy(),
                                                         self.num_bits)
        return ActivationUniformInferableQuantizer(num_bits=self.num_bits,
                                                   # In activation quantization is per-tensor only - thus we pass
                                                   # the min/max as lists with a len of 1
                                                   min_range=[min_range],
                                                   max_range=[max_range])
