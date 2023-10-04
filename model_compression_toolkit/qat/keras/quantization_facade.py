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
from functools import partial

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import FOUND_TF
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from mct_quantizers import KerasActivationQuantizationHolder
from model_compression_toolkit.trainable_infrastructure import KerasTrainableQuantizationWrapper
from model_compression_toolkit.target_platform_capabilities.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.ptq.runner import ptq_runner

if FOUND_TF:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    from tensorflow.keras.models import Model

    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.core.keras.keras_implementation import KerasImplementation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL

    from model_compression_toolkit.core.keras.back2framework.keras_model_builder import KerasModelBuilder

    from model_compression_toolkit import get_target_platform_capabilities

    from model_compression_toolkit import get_target_platform_capabilities
    from model_compression_toolkit.core import common
    from model_compression_toolkit.core.common import BaseNode
    from model_compression_toolkit.constants import TENSORFLOW
    from model_compression_toolkit.core.common.framework_info import FrameworkInfo
    from model_compression_toolkit.qat.common.qat_config import is_qat_applicable
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.qat.keras.quantizer.quantization_builder import quantization_builder, \
    get_activation_quantizer_holder
    from model_compression_toolkit.qat.common.qat_config import QATConfig

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)


    def qat_wrapper(n: common.BaseNode,
                    layer: Layer,
                    qat_config: QATConfig):
        """
        A function which takes a computational graph node and a keras layer and perform the quantization wrapping
        Args:
            qat_config: Configuration of QAT (such as training methods for example).
            n: A node of mct graph.
            layer: A keras layer.

        Returns: Wrapped layer

        """
        if is_qat_applicable(n, DEFAULT_KERAS_INFO):
            weights_quantizers, _ = quantization_builder(n,
                                                         qat_config,
                                                         DEFAULT_KERAS_INFO)
            if len(weights_quantizers) > 0:
                layer.trainable = True
                return KerasTrainableQuantizationWrapper(layer, weights_quantizers)
        return layer


    def keras_quantization_aware_training_init(in_model: Model,
                                               representative_data_gen: Callable,
                                               target_kpi: KPI = None,
                                               core_config: CoreConfig = CoreConfig(),
                                               qat_config: QATConfig = QATConfig(),
                                               fw_info: FrameworkInfo = DEFAULT_KERAS_INFO,
                                               target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_KERAS_TPC):
        """
         Prepare a trained Keras model for quantization aware training. First the model quantization is optimized
         with post-training quantization, then the model layers are wrapped with QuantizeWrappers. The model is
         quantized using a symmetric quantization thresholds (power of two).
         The model is first optimized using several transformations (e.g. BatchNormalization folding to
         preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
         being collected for each layer's output (and input, depends on the quantization configuration).
         For each possible bit width (per layer) a threshold is then being calculated using the collected
         statistics. Then, if given a mixed precision config in the core_config, using an ILP solver we find
         a mixed-precision configuration, and set a bit-width for each layer. The model is built with fake_quant
         nodes for quantizing activation. Weights are kept as float and are quantized online while training by the
         quantization wrapper's weight quantizer.
         In order to limit the maximal model's size, a target KPI need to be passed after weights_memory
         is set (in bytes).

         Args:
             in_model (Model): Keras model to quantize.
             representative_data_gen (Callable): Dataset used for initial calibration.
             target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
             core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
             qat_config (QATConfig): QAT configuration
             fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).  `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/keras/default_framework_info.py>`_
             target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Keras model according to.

         Returns:

             A quantized model.
             User information that may be needed to handle the quantized model.
             Custom-Objects dictionary for loading the saved kers model.

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
             The candidates bitwidth for quantization should be defined in the target platform model:

             >>> config = mct.core.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())

             For mixed-precision set a target KPI object:
             Create a KPI object to limit our returned model's size. Note that this value affects only coefficients
             that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
             while the bias will not):

             >>> kpi = mct.core.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

             Pass the model, the representative dataset generator, the configuration and the target KPI to get a
             quantized model:

             >>> quantized_model, quantization_info, custom_objects = mct.qat.keras_quantization_aware_training_init(model, repr_datagen, kpi, core_config=config)

             Use the quantized model for fine-tuning. For loading the model from file, use the custom_objects dictionary:

             >>> quantized_model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)

             For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

         """
        KerasModelValidation(model=in_model,
                             fw_info=fw_info).validate()

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use keras_post_training_quantization API,"
                                    "or pass a valid mixed precision configuration.")

            Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = KerasImplementation()

        # Ignore hessian service since is not used in QAT at the moment
        tg, bit_widths_config, _ = core_runner(in_model=in_model,
                                               representative_data_gen=representative_data_gen,
                                               core_config=core_config,
                                               fw_info=fw_info,
                                               fw_impl=fw_impl,
                                               tpc=target_platform_capabilities,
                                               target_kpi=target_kpi,
                                               tb_w=tb_w)

        tg = ptq_runner(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)

        _qat_wrapper = partial(qat_wrapper, qat_config=qat_config)
        qat_model, user_info = KerasModelBuilder(graph=tg,
                                                 fw_info=fw_info,
                                                 wrapper=_qat_wrapper,
                                                 get_activation_quantizer_holder_fn=partial(get_activation_quantizer_holder,
                                                                                            qat_config=qat_config)).build_model()

        user_info.mixed_precision_cfg = bit_widths_config
        #TODO: remove the last output after updating documentation.
        return qat_model, user_info, {}


    def keras_quantization_aware_training_finalize(in_model: Model) -> Model:
        """
         Convert a model fine-tuned by the user (Trainable quantizers) to a model with Inferable quantizers.

         Args:
             in_model (Model): Keras model to replace TrainableQuantizer with InferableQuantizer

         Returns:
             A quantized model with Inferable quantizers

         Examples:

             Import MCT:

             >>> import model_compression_toolkit as mct

             Import a Keras model:

             >>> from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
             >>> model = MobileNetV2()

             Create a random dataset generator:

             >>> import numpy as np
             >>> def repr_datagen(): yield [np.random.random((1, 224, 224, 3))]

             Create a MCT core config, containing the quantization configuration:

             >>> config = mct.core.CoreConfig()

             If mixed precision is desired, create a MCT core config with a mixed-precision configuration, to quantize a model with different bitwidths for different layers.
             The candidates bitwidth for quantization should be defined in the target platform model:

             >>> config = mct.core.CoreConfig(mixed_precision_config=MixedPrecisionQuantizationConfigV2())

             For mixed-precision set a target KPI object:
             Create a KPI object to limit our returned model's size. Note that this value affects only coefficients
             that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
             while the bias will not):

             >>> kpi = mct.core.KPI(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

             Pass the model, the representative dataset generator, the configuration and the target KPI to get a
             quantized model:

             >>> quantized_model, quantization_info, custom_objects = mct.qat.keras_quantization_aware_training_init(model, repr_datagen, kpi, core_config=config)

             Use the quantized model for fine-tuning. For loading the model from file, use the custom_objects dictionary:

             >>> quantized_model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
             >>> quantized_model = mct.qat.keras_quantization_aware_training_finalize(quantized_model)

         """
        def _export(layer):
            if isinstance(layer, KerasTrainableQuantizationWrapper):
                layer = layer.convert_to_inferable_quantizers()
            # In the KerasActivationQuantizationHolder case - converting the quantizers only
            # is not enough. We need to create a new layer with inferable quantizers. The reason for that
            # is that if we only convert the quantizers, the layer will have some weights (such as min, max,
            # threshold) that do not match the configuration, thus loading such a model will fail.
            # To overcome this, the convert_to_inferable_quantizers of KerasActivationQuantizationHolder
            # creates a new layer from its new configuration after converting the trainable quantizer
            # to an inferable quantizer.
            elif isinstance(layer, KerasActivationQuantizationHolder):
                layer = layer.convert_to_inferable_quantizers()
            return layer

        # clone each layer in the model and apply _export to layers with TrainableQuantizeWrappers
        exported_model = tf.keras.models.clone_model(in_model, input_tensors=None, clone_function=_export)

        return exported_model

else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def keras_quantization_aware_training_init(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_quantization_aware_training_init. '
                        'Could not find Tensorflow package.')  # pragma: no cover


    def keras_quantization_aware_training_finalize(*args, **kwargs):
        Logger.critical('Installing tensorflow is mandatory '
                        'when using keras_quantization_aware_training_finalize. '
                        'Could not find Tensorflow package.')  # pragma: no cover
