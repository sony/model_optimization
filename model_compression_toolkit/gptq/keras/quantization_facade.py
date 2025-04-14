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

from typing import Callable, Tuple, Union, Optional
from packaging import version

from model_compression_toolkit.core.common.visualization.tensorboard_writer import init_tensorboard_writer
from model_compression_toolkit.gptq.common.gptq_constants import REG_DEFAULT, LR_DEFAULT, LR_REST_DEFAULT, \
    LR_BIAS_DEFAULT, GPTQ_MOMENTUM, REG_DEFAULT_SLA
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import TENSORFLOW, ACT_HESSIAN_DEFAULT_BATCH_SIZE, GPTQ_HESSIAN_NUM_SAMPLES
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2keras import \
    AttachTpcToKeras
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.verify_packages import FOUND_TF
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfig, GPTQHessianScoresConfig, \
    GradualActivationQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import MixedPrecisionQuantizationConfig
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.runner import core_runner
from model_compression_toolkit.gptq.runner import gptq_runner
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.metadata import create_model_metadata


if FOUND_TF:
    import tensorflow as tf
    from model_compression_toolkit.core.keras.default_framework_info import DEFAULT_KERAS_INFO
    from model_compression_toolkit.gptq.keras.gptq_keras_implementation import GPTQKerasImplemantation
    from model_compression_toolkit.core.keras.keras_model_validation import KerasModelValidation
    from tensorflow.keras.models import Model
    from model_compression_toolkit.gptq.keras.gptq_loss import GPTQMultipleTensorsLoss, sample_layer_attention_loss
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.exporter.model_wrapper import get_exportable_keras_model
    from model_compression_toolkit import get_target_platform_capabilities
    from mct_quantizers.keras.metadata import add_metadata

    # As from TF2.9 optimizers package is changed
    if version.parse(tf.__version__) < version.parse("2.9"):
        from keras.optimizer_v2.optimizer_v2 import OptimizerV2  # pragma: no cover
    elif version.parse(tf.__version__) < version.parse("2.12"):
        from keras.optimizers.optimizer_v2.optimizer_v2 import OptimizerV2  # pragma: no cover
    else:
        from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

    DEFAULT_KERAS_TPC = get_target_platform_capabilities(TENSORFLOW, DEFAULT_TP_MODEL)

    def get_keras_gptq_config(n_epochs: int,
                              optimizer: OptimizerV2 = None,
                              optimizer_rest: OptimizerV2 = None,
                              loss: Callable = None,
                              log_function: Callable = None,
                              use_hessian_based_weights: bool = True,
                              regularization_factor: float = None,
                              hessian_batch_size: int = ACT_HESSIAN_DEFAULT_BATCH_SIZE,
                              use_hessian_sample_attention: bool = True,
                              gradual_activation_quantization: Union[bool, GradualActivationQuantizationConfig] = True) -> GradientPTQConfig:
        """
        Create a GradientPTQConfig instance for Keras models.

        args:
            n_epochs (int): Number of epochs for running the representative dataset for fine-tuning.
            optimizer (OptimizerV2): Keras optimizer to use for fine-tuning for auxiliry variable with a default learning rate set to 0.2.
            optimizer_rest (OptimizerV2): Keras optimizer to use for fine-tuning of the bias variable.
            loss (Callable): loss to use during fine-tuning. should accept 4 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors, the 3rd is a list of quantized weights and the 4th is a list of float weights.
            log_function (Callable): Function to log information about the gptq process.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.
            regularization_factor (float): A floating point number that defines the regularization factor.
            hessian_batch_size (int): Batch size for Hessian computation in Hessian-based weights GPTQ.
            use_hessian_sample_attention (bool): whether to use Sample-Layer Attention score for weighted loss.
            gradual_activation_quantization (bool, GradualActivationQuantizationConfig): If False, GradualActivationQuantization is disabled. If True, GradualActivationQuantization is enabled with the default settings. GradualActivationQuantizationConfig object can be passed to use non-default settings.

        returns:
            a GradientPTQConfig object to use when fine-tuning the quantized model using gptq.

        Examples:

            Import MCT and TensorFlow:

            >>> import model_compression_toolkit as mct
            >>> import tensorflow as tf

            Create a GradientPTQConfig to run for 5 epochs:

            >>> gptq_conf = mct.gptq.get_keras_gptq_config(n_epochs=5)

            Other Tensorflow optimizers can be passed:

            >>> gptq_conf = mct.gptq.get_keras_gptq_config(n_epochs=3, optimizer=tf.keras.optimizers.Nadam())

            The configuration can be passed to :func:`~model_compression_toolkit.keras_gradient_post_training_quantization` in order to quantize a keras model using gptq.


        """
        optimizer = optimizer or tf.keras.optimizers.Adam(learning_rate=LR_DEFAULT)
        optimizer_rest = optimizer_rest or tf.keras.optimizers.Adam(learning_rate=LR_REST_DEFAULT)
        bias_optimizer = tf.keras.optimizers.SGD(learning_rate=LR_BIAS_DEFAULT, momentum=GPTQ_MOMENTUM)

        if regularization_factor is None:
            regularization_factor = REG_DEFAULT_SLA if use_hessian_sample_attention else REG_DEFAULT

        hessian_weights_config = None
        if use_hessian_sample_attention:
            if not use_hessian_based_weights:    # pragma: no cover
                raise ValueError('use_hessian_based_weights must be set to True in order to use Sample Layer Attention.')

            hessian_weights_config = GPTQHessianScoresConfig(per_sample=True,
                                                             hessians_num_samples=None,
                                                             hessian_batch_size=hessian_batch_size)
            loss = loss or sample_layer_attention_loss
        elif use_hessian_based_weights:
            hessian_weights_config = GPTQHessianScoresConfig(per_sample=False,
                                                             hessians_num_samples=GPTQ_HESSIAN_NUM_SAMPLES,
                                                             hessian_batch_size=hessian_batch_size)
        
        # If a loss was not passed (and was not initialized due to use_hessian_sample_attention), use the default loss
        loss = loss or GPTQMultipleTensorsLoss()
        
        if isinstance(gradual_activation_quantization, bool):
            gradual_quant_config = GradualActivationQuantizationConfig() if gradual_activation_quantization else None
        elif isinstance(gradual_activation_quantization, GradualActivationQuantizationConfig):
            gradual_quant_config = gradual_activation_quantization
        else:
            raise TypeError(f'gradual_activation_quantization argument should be bool or '
                            f'GradualActivationQuantizationConfig, received {type(gradual_activation_quantization)}')

        return GradientPTQConfig(n_epochs=n_epochs,
                                 optimizer=optimizer,
                                 optimizer_rest=optimizer_rest,
                                 loss=loss,
                                 log_function=log_function,
                                 train_bias=True,
                                 optimizer_bias=bias_optimizer,
                                 regularization_factor=regularization_factor,
                                 hessian_weights_config=hessian_weights_config,
                                 gradual_activation_quantization_config=gradual_quant_config)


    def keras_gradient_post_training_quantization(in_model: Model, representative_data_gen: Callable,
                                                  gptq_config: GradientPTQConfig,
                                                  gptq_representative_data_gen: Callable = None,
                                                  target_resource_utilization: ResourceUtilization = None,
                                                  core_config: CoreConfig = CoreConfig(),
                                                  target_platform_capabilities: Union[TargetPlatformCapabilities, str]
                                                  = DEFAULT_KERAS_TPC) -> Tuple[Model, Optional[UserInformation]]:
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
        In order to limit the maximal model's size, a target resource utilization need to be passed after weights_memory
        is set (in bytes).
        Then, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized models, and minimizing the observed
        loss.

        Args:
            in_model (Model): Keras model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            gptq_config (GradientPTQConfig): Configuration for using gptq (e.g. optimizer).
            gptq_representative_data_gen (Callable): Dataset used for GPTQ training. If None defaults to representative_data_gen
            target_resource_utilization (ResourceUtilization): ResourceUtilization object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            target_platform_capabilities (Union[TargetPlatformCapabilities, str]): TargetPlatformCapabilities to optimize the Keras model according to.

        Returns:

            A quantized model and information the user may need to handle the quantized model.

        Examples:

            Import a Keras model:

            >>> from tensorflow.keras.applications.mobilenet import MobileNet
            >>> model = MobileNet()

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 224, 224, 3))]

            Create an MCT core config, containing the quantization configuration:

            >>> config = mct.core.CoreConfig()

            If mixed precision is desired, create an MCT core config with a mixed-precision configuration, to quantize a model
            with different bitwidths for different layers.
            The candidates bitwidth for quantization should be defined in the target platform model:

            >>> config = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=1))

            For mixed-precision set a target resource utilization object:
            Create a resource utilization object to limit our returned model's size. Note that this value affects only coefficients
            that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value,
            while the bias will not):

            >>> ru = mct.core.ResourceUtilization(model.count_params() * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

            Create GPTQ config:

            >>> gptq_config = mct.gptq.get_keras_gptq_config(n_epochs=1)

            Pass the model with the representative dataset generator to get a quantized model:

            >>> quantized_model, quantization_info = mct.gptq.keras_gradient_post_training_quantization(model, repr_datagen, gptq_config, target_resource_utilization=ru, core_config=config)

        """

        if core_config.debug_config.bypass:
            return in_model, None

        KerasModelValidation(model=in_model,
                             fw_info=DEFAULT_KERAS_INFO).validate()

        if core_config.is_mixed_precision_enabled:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
                Logger.critical("Given quantization config for mixed-precision is not of type 'MixedPrecisionQuantizationConfig'. "
                                "Ensure usage of the correct API for keras_post_training_quantization "
                                "or provide a valid mixed-precision configuration.")  # pragma: no cover

        tb_w = init_tensorboard_writer(DEFAULT_KERAS_INFO)

        fw_impl = GPTQKerasImplemantation()

        target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
        # Attach tpc model to framework
        attach2keras = AttachTpcToKeras()
        framework_platform_capabilities = attach2keras.attach(
            target_platform_capabilities,
            custom_opset2layer=core_config.quantization_config.custom_tpc_opset_to_layer)

        tg, bit_widths_config, hessian_info_service, scheduling_info = core_runner(in_model=in_model,
                                                                                   representative_data_gen=representative_data_gen,
                                                                                   core_config=core_config,
                                                                                   fw_info=DEFAULT_KERAS_INFO,
                                                                                   fw_impl=fw_impl,
                                                                                   fqc=framework_platform_capabilities,
                                                                                   target_resource_utilization=target_resource_utilization,
                                                                                   tb_w=tb_w,
                                                                                   running_gptq=True)

        float_graph = copy.deepcopy(tg)

        tg_gptq = gptq_runner(tg,
                              core_config,
                              gptq_config,
                              representative_data_gen,
                              gptq_representative_data_gen if gptq_representative_data_gen else representative_data_gen,
                              DEFAULT_KERAS_INFO,
                              fw_impl,
                              tb_w,
                              hessian_info_service=hessian_info_service)

        del hessian_info_service

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen,
                                        tb_w,
                                        float_graph,
                                        tg_gptq,
                                        fw_impl,
                                        DEFAULT_KERAS_INFO)

        exportable_model, user_info = get_exportable_keras_model(tg_gptq)
        if framework_platform_capabilities.tpc.add_metadata:
            exportable_model = add_metadata(exportable_model,
                                            create_model_metadata(fqc=framework_platform_capabilities,
                                                                  scheduling_info=scheduling_info))
        return exportable_model, user_info

else:
    # If tensorflow is not installed,
    # we raise an exception when trying to use these functions.
    def get_keras_gptq_config(*args, **kwargs):  # pragma: no cover
        Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                        "get_keras_gptq_config. The 'tensorflow' package is missing or is "
                        "installed with a version higher than 2.15.")  # pragma: no cover


    def keras_gradient_post_training_quantization(*args, **kwargs):  # pragma: no cover
        Logger.critical("Tensorflow must be installed with a version of 2.15 or lower to use "
                        "keras_gradient_post_training_quantization. The 'tensorflow' package is missing or is "
                        "installed with a version higher than 2.15.")
