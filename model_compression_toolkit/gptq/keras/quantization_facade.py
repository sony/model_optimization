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

from typing import Callable, Tuple

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.mixed_precision.kpi import KPI
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit import CoreConfig
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.gptq.runner import gptq_runner
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.core.common.target_platform.targetplatform2framework import TargetPlatformCapabilities

if common.constants.FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.gptq.keras.gptq_loss import multiple_tensors_mse_loss
    from keras.optimizer_v2.optimizer_v2 import OptimizerV2
    from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL

    from model_compression_toolkit import get_target_platform_capabilities

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def get_keras_gptq_config(n_iter: int,
                              optimizer: OptimizerV2 = tf.keras.optimizers.Adam(),
                              optimizer_rest: OptimizerV2 = tf.keras.optimizers.Adam(),
                              loss: Callable = multiple_tensors_mse_loss,
                              log_function: Callable = None,
                              train_bias: bool = True) -> GradientPTQConfig:
        """
        Create a GradientPTQConfig instance for Keras models.

        args:
            n_iter (int): Number of iterations to fine-tune.
            optimizer (OptimizerV2): Keras optimizer to use for fine-tuning for auxiliry variable.
            optimizer_rest (OptimizerV2): Keras optimizer to use for fine-tuning of the bias variable.
            loss (Callable): loss to use during fine-tuning. should accept 4 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors, the 3rd is a list of quantized weights and the 4th is a list of float weights.
            log_function (Callable): Function to log information about the gptq process.
            train_bias (bool): Whether to update the bias during the the fine-tuning or not.

        returns:
            a GradientPTQConfig object to use when fine-tuning the quantized model using gptq.

        Examples:

            Create a GradientPTQConfig to run for 5 iteration:

            >>> gptq_conf = get_keras_gptq_config(n_iter=5)

            To disable the biases training, one may set train_bias to false (enabled by default):

            >>> gptq_conf = get_keras_gptq_config(n_iter=5, train_bias=false)

            Other Tensorflow optimizers can be passed:

            >>> gptq_conf = get_keras_gptq_config(n_iter=3, optimizer=tf.keras.optimizers.Nadam())

            The configuration can be passed to :func:`~model_compression_toolkit.keras_post_training_quantization` in order to quantize a keras model using gptq.

        """

        return GradientPTQConfig(n_iter,
                                 optimizer,
                                 optimizer_rest=optimizer_rest,
                                 loss=loss,
                                 log_function=log_function,
                                 train_bias=train_bias)


    def keras_gradient_post_training_quantization_experimental(in_model: Model,
                                                               representative_data_gen: Callable,
                                                               gptq_config: GradientPTQConfig,
                                                               target_kpi: KPI = None,
                                                               core_config: CoreConfig = CoreConfig(),
                                                               fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                                               target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC) -> Tuple[Model, UserInformation]:
        """
        Quantize a trained Keras model using post-training quantization. The model is quantized using a
        symmetric constraint quantization thresholds (power of two).
        The model is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        For each possible bit width (per layer) a threshold is then being calculated using the collected
        statistics. Then, if given a mixed precision config in the core_config, using an ILP solver we find
        a mixed-precision configuration, and set a bit-width for each layer. The model is then quantized
        (both coefficients and activations by default).
        In order to limit the maximal model's size, a target KPI need to be passed after weights_memory
        is set (in bytes).
        Then, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized models, and minimizing the observed
        loss.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            gptq_config (GradientPTQConfig): Configuration for using gptq (e.g. optimizer).
            target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to.

        Returns:

            A quantized model and information the user may need to handle the quantized model.

        Examples:

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import mct:

            >>> import model_compression_toolkit as mct

            Create an MCT core config, containing the quantization configuration:

            >>> config = mct.CoreConfig()

            If mixed precision is desired, create an MCT core config with a mixed-precision configuration, to quantize a model
            with different bitwidths for different layers.
            The candidates bitwidth for quantization should be defined in the target platform model:

            >>> config = mct.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())

            For mixed-precision set a target KPI object:
            Create a KPI object to limit our returned model's size. Note that this value affects only coefficients
            that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
            while the bias will not):

            >>> kpi = mct.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

            Create GPTQ config:

            >>> gptq_config = get_keras_gptq_config(2000)

            Pass the model with the representative dataset generator to get a quantized model:

            >>> quantized_model, quantization_info = mct.keras_post_training_quantization(model, repr_datagen, gptq_config, target_kpi=kpi, core_config=config)

        """
        KerasModelValidation(model=in_model,
                             fw_info=fw_info).validate()

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                common.Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use keras_post_training_quantization API,"
                                    "or pass a valid mixed precision configuration.")

            common.Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = KerasImplementation()

        tg, bit_widths_config = core_runner(in_model=in_model,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=fw_info,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            target_kpi=target_kpi,
                                            tb_w=tb_w)

        tg_gptq = gptq_runner(tg, gptq_config, representative_data_gen,
                              fw_info, fw_impl, tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, tg_gptq, fw_impl, fw_info)

        quantized_model, user_info = export_model(tg_gptq, fw_info, fw_impl, tb_w, bit_widths_config)

        return quantized_model, user_info

else:
    # If tensorflow or tensorflow_model_optimization are not installed,
    # we raise an exception when trying to use these functions.
    def get_keras_gptq_config(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_post_training_quantization_mixed_precision. '
                        'Could not find Tensorflow package.')


    def keras_gradient_post_training_quantization_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_gradient_post_training_quantization_experimental. '
                        'Could not find Tensorflow package.')
