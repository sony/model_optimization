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

from typing import Callable

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import TENSORFLOW, FOUND_TF
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.ptq.runner import ptq_runner

if FOUND_TF:
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.exporter.model_wrapper import get_exportable_keras_model

    from model_compression_toolkit import get_target_platform_capabilities
    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def keras_post_training_quantization_experimental(in_model: Model,
                                                      representative_data_gen: Callable,
                                                      target_kpi: KPI = None,
                                                      core_config: CoreConfig = CoreConfig(),
                                                      target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC,
                                                      new_experimental_exporter: bool = True):
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

         Args:
             in_model (Model): Keras model to quantize.
             representative_data_gen (Callable): Dataset used for calibration.
             target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
             core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
             target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to.
             new_experimental_exporter (bool): Whether to wrap the quantized model using quantization information or not. Enabled by default. Experimental and subject to future changes.

         Returns:

             A quantized model and information the user may need to handle the quantized model.

         Examples:

            Import MCT:

            >>> import model_compression_toolkit as mct

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
            >>> model = MobileNetV2()

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 224, 224, 3))]

            Create a MCT core config, containing the quantization configuration:

            >>> config = mct.core.CoreConfig()

            If mixed precision is desired, create a MCT core config with a mixed-precision configuration, to quantize a model with different bitwidths for different layers.
            The candidates bitwidth for quantization should be defined in the target platform model.
            In this example we use 1 image to search mixed-precision configuration:

            >>> config = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfigV2(num_of_images=1))

            For mixed-precision set a target KPI object:
            Create a KPI object to limit our returned model's size. Note that this value affects only coefficients
            that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
            while the bias will not):

            >>> kpi = mct.core.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

            Pass the model, the representative dataset generator, the configuration and the target KPI to get a
            quantized model:

            >>> quantized_model, quantization_info = mct.ptq.keras_post_training_quantization_experimental(model, repr_datagen, kpi, core_config=config)

            For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

         """

        fw_info = DEFAULT_KERAS_INFO

        KerasModelValidation(model=in_model,
                             fw_info=fw_info).validate()

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use keras_post_training_quantization "
                                    "API, or pass a valid mixed precision configuration.")  # pragma: no cover

            Logger.info("Using experimental mixed-precision quantization. "
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

        tg = ptq_runner(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen,
                                        tb_w, tg,
                                        fw_impl,
                                        fw_info)

        if new_experimental_exporter:
            Logger.warning('Using new experimental wrapped and ready for export models. To '
                           'disable it, please set new_experimental_exporter to False when '
                           'calling keras_post_training_quantization_experimental. '
                           'If you encounter an issue please file a bug.')

            return get_exportable_keras_model(tg)

        return export_model(tg,
                            fw_info,
                            fw_impl,
                            tb_w,
                            bit_widths_config)



else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def keras_post_training_quantization_experimental(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_post_training_quantization_experimental. '
                        'Could not find Tensorflow package.')  # pragma: no cover
