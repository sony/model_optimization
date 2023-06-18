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
from model_compression_toolkit.core import common
from model_compression_toolkit.constants import FOUND_TORCH
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.gptq.common.gptq_config import GradientPTQConfigV2
from model_compression_toolkit.target_platform_capabilities.target_platform import TargetPlatformCapabilities
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.gptq.keras.quantization_facade import GPTQ_MOMENTUM
from model_compression_toolkit.gptq.runner import gptq_runner
from model_compression_toolkit.core.exporter import export_model
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2

LR_DEFAULT = 1e-4
LR_REST_DEFAULT = 1e-4
LR_BIAS_DEFAULT = 1e-4
LR_QUANTIZATION_PARAM_DEFAULT = 1e-4

if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.gptq.pytorch.gptq_pytorch_implementation import GPTQPytorchImplemantation
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.gptq.pytorch.gptq_loss import multiple_tensors_mse_loss
    from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_exportable_pytorch_model
    import torch
    from torch.nn import Module
    from torch.optim import Adam, Optimizer
    from model_compression_toolkit import get_target_platform_capabilities


    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)


    def get_pytorch_gptq_config(n_epochs: int,
                                optimizer: Optimizer = Adam([torch.Tensor([])], lr=LR_DEFAULT),
                                optimizer_rest: Optimizer = Adam([torch.Tensor([])], lr=LR_REST_DEFAULT),
                                loss: Callable = multiple_tensors_mse_loss,
                                log_function: Callable = None,
                                use_hessian_based_weights: bool = True) -> GradientPTQConfigV2:
        """
        Create a GradientPTQConfigV2 instance for Pytorch models.

        args:
            n_epochs (int): Number of epochs for running the representative dataset for fine-tuning.
            optimizer (Optimizer): Pytorch optimizer to use for fine-tuning for auxiliry variable.
            optimizer_rest (Optimizer): Pytorch optimizer to use for fine-tuning of the bias variable.
            loss (Callable): loss to use during fine-tuning. should accept 4 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors, the 3rd is a list of quantized weights and the 4th is a list of float weights.
            log_function (Callable): Function to log information about the gptq process.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.

        returns:
            a GradientPTQConfigV2 object to use when fine-tuning the quantized model using gptq.

        Examples:

            Import MCT and Create a GradientPTQConfigV2 to run for 5 epochs:

            >>> import model_compression_toolkit as mct
            >>> gptq_conf = mct.gptq.get_pytorch_gptq_config(n_epochs=5)

            Other PyTorch optimizers can be passed with dummy params:

            >>> import torch
            >>> gptq_conf = mct.gptq.get_pytorch_gptq_config(n_epochs=3, optimizer=torch.optim.Adam([torch.Tensor(1)]))

            The configuration can be passed to :func:`~model_compression_toolkit.pytorch_post_training_quantization` in order to quantize a pytorch model using gptq.

        """
        bias_optimizer = torch.optim.SGD([torch.Tensor([])], lr=LR_BIAS_DEFAULT, momentum=GPTQ_MOMENTUM)
        return GradientPTQConfigV2(n_epochs, optimizer, optimizer_rest=optimizer_rest, loss=loss,
                                   log_function=log_function, train_bias=True, optimizer_bias=bias_optimizer, use_hessian_based_weights=use_hessian_based_weights)


    def pytorch_gradient_post_training_quantization_experimental(model: Module,
                                                                 representative_data_gen: Callable,
                                                                 target_kpi: KPI = None,
                                                                 core_config: CoreConfig = CoreConfig(),
                                                                 gptq_config: GradientPTQConfigV2 = None,
                                                                 gptq_representative_data_gen: Callable = None,
                                                                 target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYTORCH_TPC,
                                                                 new_experimental_exporter: bool = True):
        """
        Quantize a trained Pytorch module using post-training quantization.
        By default, the module is quantized using a symmetric constraint quantization thresholds
        (power of two) as defined in the default TargetPlatformCapabilities.
        The module is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        Thresholds are then being calculated using the collected statistics and the module is quantized
        (both coefficients and activations by default).
        If gptq_config is passed, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized modules, and minimizing the
        observed loss.
        Then, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized models, and minimizing the observed
        loss.

        Args:
            model (Module): Pytorch model to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            gptq_config (GradientPTQConfigV2): Configuration for using gptq (e.g. optimizer).
            gptq_representative_data_gen (Callable): Dataset used for GPTQ training. If None defaults to representative_data_gen
            target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the PyTorch model according to.
            new_experimental_exporter (bool): Whether to wrap the quantized model using quantization information or not. Enabled by default. Experimental and subject to future changes.

        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import a Pytorch module:

            >>> from torchvision import models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 3, 224, 224))]

            Create MCT core configurations with number of calibration iterations set to 1:

            >>> config = mct.core.CoreConfig()

            Pass the module, the representative dataset generator and the configuration (optional) to get a quantized module

            >>> quantized_module, quantization_info = mct.gptq.pytorch_gradient_post_training_quantization_experimental(module, repr_datagen, core_config=config, gptq_config=gptq_conf)

        """

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use keras_post_training_quantization "
                                    "API, or pass a valid mixed precision configuration.")  # pragma: no cover

            Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(DEFAULT_PYTORCH_INFO)

        fw_impl = GPTQPytorchImplemantation()

        # ---------------------- #
        # Core Runner
        # ---------------------- #
        graph, bit_widths_config = core_runner(in_model=model,
                                               representative_data_gen=representative_data_gen,
                                               core_config=core_config,
                                               fw_info=DEFAULT_PYTORCH_INFO,
                                               fw_impl=fw_impl,
                                               tpc=target_platform_capabilities,
                                               target_kpi=target_kpi,
                                               tb_w=tb_w)

        # ---------------------- #
        # GPTQ Runner
        # ---------------------- #
        graph_gptq = gptq_runner(graph, core_config, gptq_config,
                                 representative_data_gen,
                                 gptq_representative_data_gen if gptq_representative_data_gen else representative_data_gen,
                                 DEFAULT_PYTORCH_INFO, fw_impl, tb_w)
        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen, tb_w, graph_gptq, fw_impl, DEFAULT_PYTORCH_INFO)

        # ---------------------- #
        # Export
        # ---------------------- #
        if new_experimental_exporter:
            Logger.warning('Using new experimental wrapped and ready for export models. To '
                           'disable it, please set new_experimental_exporter to False when '
                           'calling pytorch_gradient_post_training_quantization_experimental. '
                           'If you encounter an issue please file a bug.')

            return get_exportable_pytorch_model(graph_gptq)

        return export_model(graph_gptq,
                            DEFAULT_PYTORCH_INFO,
                            fw_impl,
                            tb_w,
                            bit_widths_config)

else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def get_pytorch_gptq_config(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_gradient_post_training_quantization_experimental. '
                        'Could not find torch package.')  # pragma: no cover


    def pytorch_gradient_post_training_quantization_experimental(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_gradient_post_training_quantization_experimental. '
                        'Could not find the torch package.')  # pragma: no cover
