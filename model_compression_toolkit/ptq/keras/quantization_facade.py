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
import copy

from typing import Callable, Tuple, Optional

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.common.visualization.tensorboard_writer import init_tensorboard_writer
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import TENSORFLOW
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.verify_packages import FOUND_TF
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.runner import core_runner
from model_compression_toolkit.ptq.runner import ptq_runner
from model_compression_toolkit.metadata import create_model_metadata

if FOUND_TF:
    from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
        AttachTpcToKeras
    from model_compression_toolkit.core.keras.default_framework_info import set_keras_info
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.exporter.model_wrapper import get_exportable_keras_model

    from model_compression_toolkit import get_target_platform_capabilities
    from mct_quantizers.keras.metadata import add_metadata

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    @set_keras_info
    def keras_post_training_quantization(in_model: Model,
                                         representative_data_gen: Callable,
                                         target_resource_utilization: ResourceUtilization = None,
                                         core_config: CoreConfig = CoreConfig(),
                                         target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC
                                         ) -> Tuple[Model, Optional[UserInformation]]:
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
         In order to limit the maximal model's size, a target ResourceUtilization need to be passed after weights_memory
         is set (in bytes).

         Args:
             in_model (Model): Keras model to quantize.
             representative_data_gen (Callable): Dataset used for calibration.
             target_resource_utilization (ResourceUtilization): ResourceUtilization object to limit the search of the mixed-precision configuration as desired.
             core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
             target_platform_capabilities (Union[TargetPlatformCapabilities, str]): TargetPlatformCapabilities to optimize the Keras model according to.

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

            >>> config = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=1))

            For mixed-precision set a target ResourceUtilization object:
            Create a ResourceUtilization object to limit our returned model's size. Note that this value affects only coefficients
            that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
            while the bias will not):

            >>> ru = mct.core.ResourceUtilization(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

            Pass the model, the representative dataset generator, the configuration and the target resource utilization to get a
            quantized model:

            >>> quantized_model, quantization_info = mct.ptq.keras_post_training_quantization(model, repr_datagen, ru, core_config=config)

            For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

         """

        if core_config.debug_config.bypass:
            return in_model, None

        KerasModelValidation(model=in_model).validate()

        if core_config.is_mixed_precision_enabled:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
                Logger.critical("Given quantization config to mixed-precision facade is not of type "
                                "MixedPrecisionQuantizationConfig. Please use keras_post_training_quantization "
                                "API, or pass a valid mixed precision configuration.")  # pragma: no cover

        tb_w = init_tensorboard_writer()

        fw_impl = KerasImplementation()

        target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
        attach2keras = AttachTpcToKeras()
        framework_platform_capabilities = attach2keras.attach(
            target_platform_capabilities,
            custom_opset2layer=core_config.quantization_config.custom_tpc_opset_to_layer)

        # Ignore returned hessian service as PTQ does not use it
        tg, bit_widths_config, _, scheduling_info = core_runner(in_model=in_model,
                                                                representative_data_gen=representative_data_gen,
                                                                core_config=core_config,
                                                                fw_impl=fw_impl,
                                                                fqc=framework_platform_capabilities,
                                                                target_resource_utilization=target_resource_utilization,
                                                                tb_w=tb_w)

        # At this point, tg is a graph that went through substitutions (such as BN folding) and is
        # ready for quantization (namely, it holds quantization params, etc.) but the weights are
        # not quantized yet. For this reason, we use it to create a graph that acts as a "float" graph
        # for things like similarity analyzer (because the quantized and float graph should have the same
        # architecture to find the appropriate compare points for similarity computation).
        similarity_baseline_graph = copy.deepcopy(tg)

        graph_with_stats_correction = ptq_runner(tg,
                                                 representative_data_gen,
                                                 core_config,
                                                 fw_impl,
                                                 tb_w)

        if core_config.debug_config.analyze_similarity:
            quantized_graph = quantize_graph_weights(graph_with_stats_correction)
            analyzer_model_quantization(representative_data_gen,
                                        tb_w,
                                        similarity_baseline_graph,
                                        quantized_graph,
                                        fw_impl)

        exportable_model, user_info = get_exportable_keras_model(graph_with_stats_correction)
        if framework_platform_capabilities.tpc.add_metadata:
            exportable_model = add_metadata(exportable_model,
                                            create_model_metadata(fqc=framework_platform_capabilities,
                                                                  scheduling_info=scheduling_info))
        return exportable_model, user_info


else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def keras_post_training_quantization(*args, **kwargs):
        Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                        "keras_post_training_quantization. The 'tensorflow' package is missing or is "
                        "installed with a version higher than 2.15.")  # pragma: no cover
