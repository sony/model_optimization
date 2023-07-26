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

import model_compression_toolkit as mct
import logging
from tutorials.quick_start.common.constants import NUM_REPRESENTATIVE_IMAGES, BATCH_SIZE, \
    REPRESENTATIVE_DATASET_FOLDER, TARGET_PLATFORM_NAME, TARGET_PLATFORM_VERSION

from model_compression_toolkit import KPI
from model_compression_toolkit.core import MixedPrecisionQuantizationConfigV2, CoreConfig
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from tutorials.quick_start.common.results import QuantInfo


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
    Calculates the model size KPI given the required weights compression

    Args:
        model: The model to calculate the KPI
        weights_compression: The required weights compression ratio
        representative_data_gen: Callable function to generate the representative dataset
        core_config (CoreConfig): CoreConfig containing parameters for quantization and mixed precision
        tpc (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the TensorFlow model according to.

    Returns:
        KPI data computed from MCT contains info about the tensors and parameters size

    """
    kpi_data = mct.core.keras_kpi_data_experimental(model, representative_data_gen, core_config=core_config, target_platform_capabilities=tpc)
    weights_kpi = 4 * kpi_data.weights_memory / weights_compression
    return KPI(weights_kpi)


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
    n_iter = math.ceil(int(args[NUM_REPRESENTATIVE_IMAGES]) // int(args[BATCH_SIZE]))
    logging.info(f"Running MCT... number of representative images: {args[REPRESENTATIVE_DATASET_FOLDER]}, number of calibration iters: {n_iter}")

    representative_data_gen = get_representative_dataset(
        representative_dataset_folder=args[REPRESENTATIVE_DATASET_FOLDER],
        n_iter=n_iter,
        batch_size=int(args[BATCH_SIZE])
    )

    # Mixed-precision configurations
    if args.get('mp_weights_compression', False):
        mp_conf = MixedPrecisionQuantizationConfigV2()
        core_conf = CoreConfig(mixed_precision_config=mp_conf)
        mp_wcr = args['mp_weights_compression']
        target_kpi = get_target_kpi(model, mp_wcr, representative_data_gen, core_conf, tpc)
    else:
        core_conf = CoreConfig()
        mp_wcr = None
        target_kpi = None

    # Quantize model
    if args.get('gptq', False):

        technique = 'GPTQ'
        n_epochs = args.get('gptq_num_calibration_iter') // n_iter
        logging.info(
            f"MCT Gradient-based Post Training Quantization is enabled. Number of epochs: {n_epochs}")

        gptq_representative_data_gen = get_representative_dataset(
            representative_dataset_folder=args[REPRESENTATIVE_DATASET_FOLDER],
            n_iter=n_iter,
            batch_size=int(args[BATCH_SIZE])
        )

        gptq_conf = mct.gptq.get_keras_gptq_config(n_epochs=n_epochs)


        quantized_model, quantization_info = \
            mct.gptq.keras_gradient_post_training_quantization_experimental(model,
                                                                              representative_data_gen=representative_data_gen,
                                                                              target_kpi=target_kpi,
                                                                              core_config=core_conf,
                                                                              gptq_config=gptq_conf,
                                                                              gptq_representative_data_gen=gptq_representative_data_gen,
                                                                              target_platform_capabilities=tpc)


    else:
        technique = 'PTQ'
        quantized_model, quantization_info = \
            mct.ptq.keras_post_training_quantization_experimental(model,
                                                                    representative_data_gen=representative_data_gen,
                                                                    target_kpi=target_kpi,
                                                                    core_config=core_conf,
                                                                    target_platform_capabilities=tpc)

    return quantized_model, QuantInfo(user_info=quantization_info, tpc_info=tpc.get_info(), technique=technique, mp_weights_compression=mp_wcr)