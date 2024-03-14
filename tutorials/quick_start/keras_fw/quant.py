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

import math
import tensorflow as tf
from keras.optimizers import Adam

import model_compression_toolkit as mct
import logging
from tutorials.quick_start.common.constants import NUM_REPRESENTATIVE_IMAGES, BATCH_SIZE, \
    REPRESENTATIVE_DATASET_FOLDER, TARGET_PLATFORM_NAME, TARGET_PLATFORM_VERSION, BYTES_TO_FP32, MP_WEIGHTS_COMPRESSION


from model_compression_toolkit.core import MixedPrecisionQuantizationConfig, CoreConfig, KPI
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from tutorials.quick_start.common.results import QuantInfo
from tutorials.quick_start.common.tpc_info import get_tpc_info


def get_tpc(target_platform_name: str, target_platform_version: str) -> TargetPlatformCapabilities:
    """
    Returns the target platform capabilities according to the given platform name and version.

    Args:
        target_platform_name: Name of the target platform
        target_platform_version: Version of the target platform

    Returns:
        The target platform capabilities.

    """
    return mct.get_target_platform_capabilities('tensorflow', target_platform_name, target_platform_version)


def get_target_kpi(model, weights_compression, representative_data_gen, core_config, tpc):
    """
    Calculates the model's required size according to the given weights compression rate, to provide as a constraint for mixed precision search.

    Args:
        model: The model to calculate the KPI.
        weights_compression: The required weights compression ratio.
        representative_data_gen: Callable function to generate the representative dataset.
        core_config (CoreConfig): CoreConfig containing parameters for quantization and mixed precision.
        tpc (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the TensorFlow model according to.

    Returns:
        A KPI object computed from MCT and contains info about the target model size.

    """
    kpi_data = mct.core.keras_kpi_data(model, representative_data_gen, core_config=core_config,
                                       target_platform_capabilities=tpc)
    weights_kpi = BYTES_TO_FP32 * kpi_data.weights_memory / weights_compression # (4 bytes for fp32) * weights memory(in Bytes) / compression rate
    return KPI(weights_memory=weights_kpi)


def quantize(model: tf.keras.Model,
             get_representative_dataset: callable,
             tpc: TargetPlatformCapabilities,
             args: dict):
    """
    Returns a quantized model and a quantization info from MCT for the given TensorFlow floating-point model.

    Args:
        model: TensorFlow floating-point model to be quantized.
        get_representative_dataset: Callable function to generate the representative dataset for quantization.
        tpc: Target platform capabilities.
        args: Dictionary containing the necessary configurations for quantization.

    Returns:
        Tuple containing the quantized model and quantization info.

    """

    # PTQ - general configurations
    n_iter = math.ceil(int(args[NUM_REPRESENTATIVE_IMAGES]) // int(args[BATCH_SIZE])) # Number of batches
    logging.info(f"Running MCT... number of representative images: {args[REPRESENTATIVE_DATASET_FOLDER]}, number of calibration iters: {n_iter}")

    representative_data_gen = get_representative_dataset(
        representative_dataset_folder=args[REPRESENTATIVE_DATASET_FOLDER],
        n_iter=n_iter,
        batch_size=int(args[BATCH_SIZE])
    )

    # Mixed-precision configurations
    mp_wcr = args.get(MP_WEIGHTS_COMPRESSION, None)
    if mp_wcr:
        mp_conf = MixedPrecisionQuantizationConfig()
        core_conf = CoreConfig(quantization_config=mct.core.QuantizationConfig(
            shift_negative_activation_correction=True),
                               mixed_precision_config=mp_conf)
        target_kpi = get_target_kpi(model, mp_wcr, representative_data_gen, core_conf, tpc)
        core_conf.mixed_precision_config.set_target_kpi(target_kpi)
    else:
        core_conf = CoreConfig(quantization_config=mct.core.QuantizationConfig(
            shift_negative_activation_correction=True))

    # Quantize model
    if args.get('gptq', False):

        workflow = 'GPTQ'
        n_epochs = args.get('gptq_num_calibration_iter') // n_iter
        logging.info(
            f"MCT Gradient-based Post Training Quantization is enabled. Number of epochs: {n_epochs}")

        gptq_conf = mct.gptq.get_keras_gptq_config(n_epochs=n_epochs, optimizer=Adam(learning_rate=args['gptq_lr']))

        quantized_model, quantization_info = \
            mct.gptq.keras_gradient_post_training_quantization(model, representative_data_gen=representative_data_gen,
                                                               gptq_config=gptq_conf,
                                                               gptq_representative_data_gen=representative_data_gen,
                                                               core_config=core_conf, target_platform_capabilities=tpc)


    else:
        workflow = 'PTQ'
        quantized_model, quantization_info = \
            mct.ptq.keras_post_training_quantization(model,
                                                     representative_data_gen=representative_data_gen,
                                                     core_config=core_conf,
                                                     target_platform_capabilities=tpc)

    return quantized_model, QuantInfo(user_info=quantization_info,
                                      tpc_info=get_tpc_info(tpc=tpc),
                                      quantization_workflow=workflow,
                                      mp_weights_compression=mp_wcr)