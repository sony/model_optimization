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
from typing import Callable, Union, Optional, Tuple

from model_compression_toolkit.constants import ACT_HESSIAN_DEFAULT_BATCH_SIZE, PYTORCH, GPTQ_HESSIAN_NUM_SAMPLES
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import \
    ResourceUtilization
from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.common.visualization.tensorboard_writer import init_tensorboard_writer
from model_compression_toolkit.core.runner import core_runner
from model_compression_toolkit.gptq.common.gptq_config import (
    GradientPTQConfig, GPTQHessianScoresConfig, GradualActivationQuantizationConfig)
from model_compression_toolkit.gptq.common.gptq_constants import REG_DEFAULT, LR_DEFAULT, LR_REST_DEFAULT, \
    LR_BIAS_DEFAULT, GPTQ_MOMENTUM, REG_DEFAULT_SLA
from model_compression_toolkit.gptq.runner import gptq_runner
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.metadata import create_model_metadata
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.verify_packages import FOUND_TORCH



if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import set_pytorch_info
    from model_compression_toolkit.gptq.pytorch.gptq_pytorch_implementation import GPTQPytorchImplemantation
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.gptq.pytorch.gptq_loss import multiple_tensors_mse_loss, sample_layer_attention_loss
    from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_exportable_pytorch_model
    import torch
    from torch.nn import Module
    from torch.optim import Adam, Optimizer
    from model_compression_toolkit import get_target_platform_capabilities
    from mct_quantizers.pytorch.metadata import add_metadata
    from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
        AttachTpcToPytorch

    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    def get_pytorch_gptq_config(n_epochs: int,
                                optimizer: Optimizer = None,
                                optimizer_rest: Optimizer = None,
                                loss: Callable = None,
                                log_function: Callable = None,
                                use_hessian_based_weights: bool = True,
                                regularization_factor: float = None,
                                hessian_batch_size: int = ACT_HESSIAN_DEFAULT_BATCH_SIZE,
                                use_hessian_sample_attention: bool = True,
                                gradual_activation_quantization: Union[bool, GradualActivationQuantizationConfig] = True,
                                ) -> GradientPTQConfig:
        """
        Create a GradientPTQConfig instance for Pytorch models.

        args:
            n_epochs (int): Number of epochs for running the representative dataset for fine-tuning.
            optimizer (Optimizer): Pytorch optimizer to use for fine-tuning for auxiliry variable.
            optimizer_rest (Optimizer): Pytorch optimizer to use for fine-tuning of the bias variable.
            loss (Callable): loss to use during fine-tuning. See the default loss function for the exact interface.
            log_function (Callable): Function to log information about the gptq process.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.
            regularization_factor (float): A floating point number that defines the regularization factor.
            hessian_batch_size (int): Batch size for Hessian computation in Hessian-based weights GPTQ.
            use_hessian_sample_attention (bool): whether to use Sample-Layer Attention score for weighted loss.
            gradual_activation_quantization (bool, GradualActivationQuantizationConfig): If False, GradualActivationQuantization is disabled. If True, GradualActivationQuantization is enabled with the default settings. GradualActivationQuantizationConfig object can be passed to use non-default settings.

        returns:
            a GradientPTQConfig object to use when fine-tuning the quantized model using gptq.

        Examples:

            Import MCT and Create a GradientPTQConfig to run for 5 epochs:

            >>> import model_compression_toolkit as mct
            >>> gptq_conf = mct.gptq.get_pytorch_gptq_config(n_epochs=5)

            Other PyTorch optimizers can be passed with dummy params:

            >>> import torch
            >>> gptq_conf = mct.gptq.get_pytorch_gptq_config(n_epochs=3, optimizer=torch.optim.Adam([torch.Tensor(1)]))

            The configuration can be passed to :func:`~model_compression_toolkit.pytorch_gradient_post_training_quantization` in order to quantize a pytorch model using gptq.

        """
        optimizer = optimizer or Adam([torch.Tensor([])], lr=LR_DEFAULT)
        optimizer_rest = optimizer_rest or Adam([torch.Tensor([])], lr=LR_REST_DEFAULT)
        # TODO this contradicts the docstring for optimizer_rest
        bias_optimizer = torch.optim.SGD([torch.Tensor([])], lr=LR_BIAS_DEFAULT, momentum=GPTQ_MOMENTUM)

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
        loss = loss or multiple_tensors_mse_loss

        if isinstance(gradual_activation_quantization, bool):
            gradual_quant_config = GradualActivationQuantizationConfig() if gradual_activation_quantization else None
        elif isinstance(gradual_activation_quantization, GradualActivationQuantizationConfig):
            gradual_quant_config = gradual_activation_quantization
        else:    # pragma: no cover
            raise TypeError(f'gradual_activation_quantization argument should be bool or '
                            f'GradualActivationQuantizationConfig, received {type(gradual_activation_quantization)}')

        return GradientPTQConfig(n_epochs=n_epochs,
                                 loss=loss,
                                 optimizer=optimizer,
                                 optimizer_rest=optimizer_rest,
                                 optimizer_bias=bias_optimizer,
                                 train_bias=True,
                                 regularization_factor=regularization_factor,
                                 hessian_weights_config=hessian_weights_config,
                                 gradual_activation_quantization_config=gradual_quant_config,
                                 log_function=log_function)


    @set_pytorch_info
    def pytorch_gradient_post_training_quantization(model: Module,
                                                    representative_data_gen: Callable,
                                                    target_resource_utilization: ResourceUtilization = None,
                                                    core_config: CoreConfig = CoreConfig(),
                                                    gptq_config: GradientPTQConfig = None,
                                                    gptq_representative_data_gen: Callable = None,
                                                    target_platform_capabilities: Union[TargetPlatformCapabilities, str] = DEFAULT_PYTORCH_TPC
                                                    ) -> Tuple[Module, Optional[UserInformation]]:
        """
        Quantize a trained Pytorch module using post-training quantization.
        By default, the module is quantized using a symmetric constraint quantization thresholds
        (power of two) as defined in the default FrameworkQuantizationCapabilities.
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
            target_resource_utilization (ResourceUtilization): ResourceUtilization object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            gptq_config (GradientPTQConfig): Configuration for using gptq (e.g. optimizer).
            gptq_representative_data_gen (Callable): Dataset used for GPTQ training. If None defaults to representative_data_gen
            target_platform_capabilities (Union[TargetPlatformCapabilities, str]): TargetPlatformCapabilities to optimize the PyTorch model according to.

        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import Model Compression Toolkit:

            >>> import model_compression_toolkit as mct

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

            >>> quantized_module, quantization_info = mct.gptq.pytorch_gradient_post_training_quantization(module, repr_datagen, core_config=config, gptq_config=gptq_conf)

        """

        if core_config.debug_config.bypass:
            return model, None

        if core_config.is_mixed_precision_enabled:    # pragma: no cover
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
                Logger.critical("Given quantization config for mixed-precision is not of type 'MixedPrecisionQuantizationConfig'. "
                                "Ensure usage of the correct API for 'pytorch_gradient_post_training_quantization' "
                                "or provide a valid mixed-precision configuration.")
        tb_w = init_tensorboard_writer()

        fw_impl = GPTQPytorchImplemantation()

        target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
        # Attach tpc model to framework
        attach2pytorch = AttachTpcToPytorch()
        framework_quantization_capabilities = attach2pytorch.attach(target_platform_capabilities,
                                                             core_config.quantization_config.custom_tpc_opset_to_layer)

        # ---------------------- #
        # Core Runner
        # ---------------------- #
        graph, bit_widths_config, hessian_info_service, scheduling_info = core_runner(in_model=model,
                                                                                      representative_data_gen=representative_data_gen,
                                                                                      core_config=core_config,
                                                                                      fw_impl=fw_impl,
                                                                                      fqc=framework_quantization_capabilities,
                                                                                      target_resource_utilization=target_resource_utilization,
                                                                                      tb_w=tb_w,
                                                                                      running_gptq=True)

        float_graph = copy.deepcopy(graph)

        # ---------------------- #
        # GPTQ Runner
        # ---------------------- #
        graph_gptq = gptq_runner(graph,
                                 core_config,
                                 gptq_config,
                                 representative_data_gen,
                                 gptq_representative_data_gen if gptq_representative_data_gen else representative_data_gen,
                                 fw_impl,
                                 tb_w,
                                 hessian_info_service=hessian_info_service)

        if core_config.debug_config.analyze_similarity:
            analyzer_model_quantization(representative_data_gen,
                                        tb_w,
                                        float_graph,
                                        graph_gptq,
                                        fw_impl)

        exportable_model, user_info = get_exportable_pytorch_model(graph_gptq)
        if framework_quantization_capabilities.tpc.add_metadata:
            exportable_model = add_metadata(exportable_model,
                                            create_model_metadata(fqc=framework_quantization_capabilities,
                                                                  scheduling_info=scheduling_info))
        return exportable_model, user_info


else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def get_pytorch_gptq_config(*args, **kwargs):    # pragma: no cover
        Logger.critical("PyTorch must be installed to use 'get_pytorch_gptq_config'. "
                        "The 'torch' package is missing.")


    def pytorch_gradient_post_training_quantization(*args, **kwargs):    # pragma: no cover
        Logger.critical("PyTorch must be installed to use 'pytorch_gradient_post_training_quantization'. "
                        "The 'torch' package is missing.")
