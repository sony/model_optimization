# Copyright 2021 Sony Semiconductor Israel, Inc. All rights reserved.
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

from typing import Callable, List, Tuple

from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.constants import TENSORFLOW
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.network_editors.actions import EditRule
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.core.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.core.common.quantization.core_config import CoreConfig
from model_compression_toolkit.core.common.quantization.debug_config import DebugConfig
from model_compression_toolkit.core.common.quantization.quantization_config import DEFAULTCONFIG
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.gptq.runner import gptq_runner
from model_compression_toolkit.ptq.runner import ptq_runner
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.analyzer import analyzer_model_quantization

import importlib

from model_compression_toolkit.core.common.target_platform.targetplatform2framework import TargetPlatformCapabilities


if importlib.util.find_spec("tensorflow") is not None\
        and importlib.util.find_spec("tensorflow_model_optimization") is not None:
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.core.keras.constants import DEFAULT_TP_MODEL

    from model_compression_toolkit import get_target_platform_capabilities
    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def keras_post_training_quantization(in_model: Model,
                                         representative_data_gen: Callable,
                                         n_iter: int = 500,
                                         quant_config: QuantizationConfig = DEFAULTCONFIG,
                                         fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                         network_editor: List[EditRule] = [],
                                         gptq_config: GradientPTQConfig = None,
                                         analyze_similarity: bool = False,
                                         target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC) -> Tuple[Model, UserInformation]:
        """
        Quantize a pretrained Keras model using post-training quantization. By default, the model is quantized
        using a symmetric constraint quantization thresholds (power of two) as defined in the default TargetPlatformCapabilities.
        The model is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        Thresholds are then being calculated using the collected statistics and the model is quantized
        (both coefficients and activations by default).
        If a gptq_config is passed, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized models, and minimizing the observed
        loss.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            n_iter (int): Number of calibration iterations to run.
            quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized. `Default configuration. <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/core/common/quantization/quantization_config.py#L154>`_
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
            network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
            gptq_config (GradientPTQConfig): Configuration for using gptq (e.g. optimizer).
            analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to. `Default Keras TPC <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/tpc_models/keras_tp_models/keras_default.py>`_

        Returns:
            A quantized model and information the user may need to handle the quantized model.

        Examples:

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import mct and pass the model with the representative dataset generator to get a quantized model:

            >>> import model_compression_toolkit as mct
            >>> quantized_model, quantization_info = mct.keras_post_training_quantization(model, repr_datagen, n_iter=1)

        """
        KerasModelValidation(model=in_model,
                             fw_info=fw_info).validate()

        core_config = CoreConfig(n_iter,
                                 quantization_config=quant_config,
                                 debug_config=DebugConfig(analyze_similarity=analyze_similarity,
                                                          network_editor=network_editor)
                                 )

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = KerasImplementation()

        tg, bit_widths_config = core_runner(in_model=in_model,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=fw_info,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            tb_w=tb_w)

        if gptq_config is None:
            tg = ptq_runner(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)
        else:
            tg = gptq_runner(tg, core_config, gptq_config, representative_data_gen,
                             fw_info, fw_impl, tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, tg, fw_impl, fw_info)

        quantized_model, user_info = export_model(tg, fw_info, fw_impl, tb_w, bit_widths_config)

        return quantized_model, user_info


    def keras_post_training_quantization_mixed_precision(in_model: Model,
                                                         representative_data_gen: Callable,
                                                         target_kpi: KPI,
                                                         n_iter: int = 500,
                                                         quant_config: MixedPrecisionQuantizationConfig = DEFAULT_MIXEDPRECISION_CONFIG,
                                                         fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                                         network_editor: List[EditRule] = [],
                                                         gptq_config: GradientPTQConfig = None,
                                                         analyze_similarity: bool = False,
                                                         target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC) -> Tuple[Model, UserInformation]:
        """
         Quantize a pretrained Keras model using post-training quantization. By default, the model is quantized
         using a symmetric constraint quantization thresholds (power of two) as defined in the default
         TargetPlatformCapabilities.
         The model is first optimized using several transformations (e.g. BatchNormalization folding to
         preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
         being collected for each layer's output (and input, depends on the quantization configuration).
         For each possible bit width (per operator, as defined in the TargetPlatformCapabilities) a
         threshold is then being calculated using the collected statistics.
         Then, using an ILP solver we find a mixed-precision configuration, and set a bit width
         for each quantizer (for both activations and weights quantizers, by default).
         In order to limit the maximal model's size, a target KPI need to be passed after weights_memory
         or activation_memory (or both) is set (in bytes).
         The model is then quantized (both coefficients and activations by default).
         If gptq_config is passed, the quantized weights are optimized using gradient based post
         training quantization by comparing points between the float and quantized models, and minimizing the
         observed loss.
         Notice that this feature is experimental.

         Args:
             in_model (Model): Keras model to quantize.
             representative_data_gen (Callable): Dataset used for calibration.
             target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
             n_iter (int): Number of calibration iterations to run.
             quant_config (MixedPrecisionQuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized.
             fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
             network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
             gptq_config (GradientPTQConfig): Configuration for using GPTQ (e.g. optimizer).
             analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
             target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to. `Default Keras TPC <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/tpc_models/keras_tp_models/keras_default.py>`_


         Returns:
             A quantized model and information the user may need to handle the quantized model.

         Examples:

             Import MCT:

             >>> import model_compression_toolkit as mct

             Import a Keras model:

             >>> from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
             >>> model = MobileNetV2()

             Create a random dataset generator:

             >>> import numpy as np
             >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

             Create a mixed-precision configuration, to quantize a model with different bitwidths for different layers.
             The candidates bitwidth for quantization should be defined in the target platform model:

             >>> config = mct.MixedPrecisionQuantizationConfig()

             Create a KPI object to limit our returned model's size. Note that this value affects only coefficients
             that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
             while the bias will not):

             >>> kpi = mct.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

             Pass the model, the representative dataset generator, the configuration and the target KPI to get a
             quantized model:

             >>> quantized_model, quantization_info = mct.keras_post_training_quantization_mixed_precision(model,repr_datagen, target_kpi=kpi, n_iter=10, quant_config=config)

             For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/experimental_api_docs/modules/mixed_precision_quantization_config.html#model_compression_toolkit.MixedPrecisionQuantizationConfigV2>`_.

         """
        KerasModelValidation(model=in_model,
                             fw_info=fw_info).validate()

        if not isinstance(quant_config, MixedPrecisionQuantizationConfig):
            common.Logger.error("Given quantization config to mixed-precision facade is not of type "
                                "MixedPrecisionQuantizationConfig. Please use keras_post_training_quantization API,"
                                "or pass a valid mixed precision configuration.")

        common.Logger.info("Using experimental mixed-precision quantization. "
                           "If you encounter an issue please file a bug.")

        quantization_config, mp_config = quant_config.separate_configs()
        core_config = CoreConfig(n_iter,
                                 quantization_config=quantization_config,
                                 mixed_precision_config=mp_config,
                                 debug_config=DebugConfig(analyze_similarity=analyze_similarity,
                                                          network_editor=network_editor)
                                 )

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

        if gptq_config is None:
            tg = ptq_runner(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)
        else:
            tg = gptq_runner(tg, core_config, gptq_config, representative_data_gen,
                             fw_info, fw_impl, tb_w)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, tg, fw_impl, fw_info)

        quantized_model, user_info = export_model(tg, fw_info, fw_impl, tb_w, bit_widths_config)

        return quantized_model, user_info

else:
    # If tensorflow or tensorflow_model_optimization are not installed,
    # we raise an exception when trying to use these functions.
    def keras_post_training_quantization(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_post_training_quantization. '
                        'Could not find Tensorflow package.')

    def keras_post_training_quantization_mixed_precision(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using keras_post_training_quantization_mixed_precision. '
                        'Could not find Tensorflow package.')
