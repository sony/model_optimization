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
from typing import Callable, List

from model_compression_toolkit import common
from model_compression_toolkit.common import Logger
from model_compression_toolkit.common.constants import PYTORCH
from model_compression_toolkit.common.gptq.gptq_config import GradientPTQConfig
from model_compression_toolkit.common.hardware_representation import FrameworkHardwareModel
from model_compression_toolkit.common.mixed_precision.kpi import KPI
from model_compression_toolkit.common.framework_info import FrameworkInfo
from model_compression_toolkit.common.network_editors.actions import EditRule
from model_compression_toolkit.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig, DEFAULT_MIXEDPRECISION_CONFIG
from model_compression_toolkit.common.post_training_quantization import post_training_quantization
from model_compression_toolkit.common.quantization.quantization_config import QuantizationConfig
from model_compression_toolkit.common.quantization.quantization_config import DEFAULTCONFIG

import importlib

if importlib.util.find_spec("torch") is not None:
    from model_compression_toolkit.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.pytorch.constants import DEFAULT_HWM
    from torch.nn import Module

    from model_compression_toolkit import get_model
    PYTORCH_DEFAULT_MODEL = get_model(PYTORCH, DEFAULT_HWM)

    def pytorch_post_training_quantization(in_module: Module,
                                           representative_data_gen: Callable,
                                           n_iter: int = 500,
                                           quant_config: QuantizationConfig = DEFAULTCONFIG,
                                           fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                                           network_editor: List[EditRule] = [],
                                           gptq_config: GradientPTQConfig = None,
                                           analyze_similarity: bool = False,
                                           fw_hw_model: FrameworkHardwareModel = PYTORCH_DEFAULT_MODEL):
        """
        Quantize a trained Pytorch module using post-training quantization. The module is quantized using a
        symmetric constraint quantization thresholds (power of two).
        The module is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        Thresholds are then being calculated using the collected statistics and the module is quantized
        (both coefficients and activations by default).
        If a gptq configuration is passed, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized modules, and minimizing the observed loss.

        Args:
            in_module (Module): Pytorch module to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            n_iter (int): Number of calibration iterations to run.
            quant_config (QuantizationConfig): QuantizationConfig containing parameters of how the module should be quantized. `Default configuration. <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/common/quantization/quantization_config.py#L154>`_
            fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Pytorch info <https://github.com/sony/model_optimization/blob/21e21c95ca25a31874a5be7af9dd2dd5da8f3a10/model_compression_toolkit/pytorch/default_framework_info.py#L113>`_
            network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
            gptq_config (GradientPTQConfig): Configuration for using gptq (e.g. optimizer).
            analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
            fw_hw_model (FrameworkHardwareModel): FrameworkHardwareModel to optimize the Keras model according to.


        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import a Pytorch module:

            >>> import torchvision.models.mobilenet_v2 as models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

            Import mct and pass the module with the representative dataset generator to get a quantized module:

            >>> import model_compression_toolkit as mct
            >>> quantized_module, quantization_info = mct.pytorch_post_training_quantization(module, repr_datagen)

        """

        return post_training_quantization(in_module,
                                          representative_data_gen,
                                          n_iter,
                                          quant_config,
                                          fw_info,
                                          PytorchImplementation(),
                                          fw_hw_model,
                                          network_editor,
                                          gptq_config,
                                          analyze_similarity)


    def pytorch_post_training_quantization_mixed_precision(in_model: Module,
                                                           representative_data_gen: Callable,
                                                           target_kpi: KPI,
                                                           n_iter: int = 500,
                                                           quant_config: MixedPrecisionQuantizationConfig = DEFAULT_MIXEDPRECISION_CONFIG,
                                                           fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                                                           network_editor: List[EditRule] = [],
                                                           gptq_config: GradientPTQConfig = None,
                                                           analyze_similarity: bool = False,

                                                           fw_hw_model: FrameworkHardwareModel = PYTORCH_DEFAULT_MODEL):
        """
         Quantize a trained Pytorch model using post-training quantization. The model is quantized using a
         symmetric constraint quantization thresholds (power of two).
         The model is first optimized using several transformations (e.g. BatchNormalization folding to
         preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
         being collected for each layer's output (and input, depends on the quantization configuration).
         For each possible bit width (per layer) a threshold is then being calculated using the collected
         statistics. Then, using an ILP solver we find a mixed-precision configuration, and set a bit width
         for each layer. The model is then quantized (both coefficients and activations by default).
         In order to limit the maximal model's size, a target KPI need to be passed after weights_memory
         is set (in bytes).
         If a gptq configuration is passed, the quantized weights are optimized using gradient based post
         training quantization by comparing points between the float and quantized models, and minimizing the observed loss.
         Notice that this feature is experimental.
         **For now, mixed precision is supported for weights only.**

         Args:
             in_model (Model): Pytorch model to quantize.
             representative_data_gen (Callable): Dataset used for calibration.
             target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
             n_iter (int): Number of calibration iterations to run.
             quant_config (MixedPrecisionQuantizationConfig): QuantizationConfig containing parameters of how the model should be quantized.
             fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.). `Default Keras info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/pytorch/default_framework_info.py#L100>`_
             network_editor (List[EditRule]): List of EditRules. Each EditRule consists of a node filter and an action to change quantization settings of the filtered nodes.
             gptq_config (GradientPTQConfig): Configuration for using GPTQ (e.g. optimizer).
             analyze_similarity (bool): Whether to plot similarity figures within TensorBoard (when logger is enabled) or not.
             fw_hw_model (FrameworkHardwareModel): FrameworkHardwareModel to optimize the Keras model according to.

         Returns:
             A quantized model and information the user may need to handle the quantized model.

         Examples:

             Import MCT:

             >>> import model_compression_toolkit as mct

             Import a Pytorch model:

             >>> import torchvision.models.mobilenet_v2 as models
             >>> module = models.mobilenet_v2()

             Create a random dataset generator:

             >>> import numpy as np
             >>> def repr_datagen(): return [np.random.random((1,224,224,3))]

             Create a mixed-precision configuration, to quantize a model with different bitwidths for different layers.
             The candidates bitwidth for quantization should be defined in the hardware model:

             >>> config = mct.MixedPrecisionQuantizationConfig()

             Create a KPI object to limit our returned model's size. Note that this value affects only coefficients that should be quantized (for example, the kernel of Conv2D in Keras will be affected by this value, while the bias will not):

             >>> kpi = mct.KPI(sum(p.numel() for p in module.parameters()) * 0.75)  # About 0.75 of the model size when quantized with 8 bits.

             Pass the model, the representative dataset generator, the configuration and the target KPI to get a quantized model:

             >>> quantized_model, quantization_info = mct.pytorch_post_training_quantization_mixed_precision(module, repr_datagen, n_iter=10, quant_config=config, target_kpi=kpi)

             For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

         """
        if not isinstance(quant_config, MixedPrecisionQuantizationConfig):
            common.Logger.error("Given quantization config to mixed-precision facade is not of type "
                                "MixedPrecisionQuantizationConfig. Please use pytorch_post_training_quantization API, "
                                "or pass a valid mixed precision configuration.")

        common.Logger.info("Using experimental mixed-precision quantization. "
                           "If you encounter an issue please file a bug.")

        return post_training_quantization(in_model,
                                          representative_data_gen,
                                          n_iter,
                                          quant_config,
                                          fw_info,
                                          PytorchImplementation(),
                                          fw_hw_model,
                                          network_editor,
                                          gptq_config,
                                          analyze_similarity,
                                          target_kpi)

else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_post_training_quantization(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_post_training_quantization. '
                        'Could not find the torch package.')

    def pytorch_post_training_quantization_mixed_precision(*args, **kwargs):
        Logger.critical('Installing tensorflow and tensorflow_model_optimization is mandatory '
                        'when using pytorch_post_training_quantization_mixed_precision. '
                        'Could not find Tensorflow package.')
