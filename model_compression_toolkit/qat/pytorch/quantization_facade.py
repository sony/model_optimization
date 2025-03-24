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
from typing import Callable, Union
from functools import partial

from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
    AttachTpcToPytorch
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.verify_packages import FOUND_TORCH

from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common.visualization.tensorboard_writer import init_tensorboard_writer
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.runner import core_runner
from model_compression_toolkit.ptq.runner import ptq_runner

if FOUND_TORCH:
    import torch.nn as nn
    from torch.nn import Module
    from mct_quantizers import PytorchActivationQuantizationHolder
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.qat.common.qat_config import is_qat_applicable
    from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
    from mct_quantizers import PytorchQuantizationWrapper
    from model_compression_toolkit import get_target_platform_capabilities
    from model_compression_toolkit.qat.common.qat_config import QATConfig
    from model_compression_toolkit.qat.pytorch.quantizer.quantization_builder import get_activation_quantizer_holder
    from model_compression_toolkit.qat.pytorch.quantizer.quantization_builder import quantization_builder

    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)


    def qat_wrapper(n: common.BaseNode,
                    module: nn.Module,
                    qat_config: QATConfig):
        """
        A function which takes a computational graph node and a pytorch module and perform the quantization wrapping
        Args:
            n: A node of mct graph.
            module: A Pytorch module
            qat_config (QATConfig): QAT configuration
        Returns: Wrapped layer

        """
        if is_qat_applicable(n, DEFAULT_PYTORCH_INFO):
            # If we are here, then the node has a kernel attribute to quantize and training during QAT
            weights_quantizers, _ = quantization_builder(n, qat_config,
                                                         DEFAULT_PYTORCH_INFO.get_kernel_op_attributes(n.type)[0])
            if len(weights_quantizers) > 0:
                return PytorchQuantizationWrapper(module, weights_quantizers)

        # TODO: need to check if in this case, if there are other weights attributes that are not trainable but are
        #  quantized, do we need to wrap them as well?
        return module


    def pytorch_quantization_aware_training_init_experimental(in_model: Module,
                                                              representative_data_gen: Callable,
                                                              target_resource_utilization: ResourceUtilization = None,
                                                              core_config: CoreConfig = CoreConfig(),
                                                              qat_config: QATConfig = QATConfig(),
                                                              target_platform_capabilities: Union[TargetPlatformCapabilities, str]
                                                              = DEFAULT_PYTORCH_TPC):
        """
         Prepare a trained Pytorch model for quantization aware training. First the model quantization is optimized
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
         In order to limit the maximal model's size, a target resource utilization need to be passed after weights_memory
         is set (in bytes).

         Args:
             in_model (Model): Pytorch model to quantize.
             representative_data_gen (Callable): Dataset used for initial calibration.
             target_resource_utilization (ResourceUtilization): ResourceUtilization object to limit the search of the mixed-precision configuration as desired.
             core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
             qat_config (QATConfig): QAT configuration
             target_platform_capabilities (Union[TargetPlatformCapabilities, str]): TargetPlatformCapabilities to optimize the Pytorch model according to.

         Returns:

             A quantized model.
             User information that may be needed to handle the quantized model.

         Examples:
             Import MCT:

             >>> import model_compression_toolkit as mct

             Import a Pytorch model:

             >>> from torchvision.models import mobilenet_v2
             >>> model = mobilenet_v2(pretrained=True)

             Create a random dataset generator, for required number of calibration iterations (num_calibration_batches). In this example, a random dataset of 10 batches each containing 4 images is used:

             >>> import numpy as np
             >>> num_calibration_batches = 10
             >>> def repr_datagen():
             >>>     for _ in range(num_calibration_batches):
             >>>         yield [np.random.random((4, 3, 224, 224))]

             Create a MCT core config, containing the quantization configuration:

             >>> config = mct.core.CoreConfig()

             Pass the model, the representative dataset generator, the configuration and the target resource utilization to get a quantized model. Now the model contains quantizer wrappers for fine tunning the weights:

             >>> quantized_model, quantization_info = mct.qat.pytorch_quantization_aware_training_init_experimental(model, repr_datagen, core_config=config)

             For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

         """
        Logger.warning(
            f"pytorch_quantization_aware_training_init_experimental is experimental and is subject to future changes."
            f"If you encounter an issue, please open an issue in our GitHub "
            f"project https://github.com/sony/model_optimization")

        if core_config.is_mixed_precision_enabled:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
                Logger.critical("Given quantization config to mixed-precision facade is not of type "
                                "MixedPrecisionQuantizationConfig. Please use pytorch_post_training_quantization API,"
                                "or pass a valid mixed precision configuration.")

        tb_w = init_tensorboard_writer(DEFAULT_PYTORCH_INFO)
        fw_impl = PytorchImplementation()

        target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
        # Attach tpc model to framework
        attach2pytorch = AttachTpcToPytorch()
        framework_platform_capabilities = attach2pytorch.attach(target_platform_capabilities,
                                                                core_config.quantization_config.custom_tpc_opset_to_layer)

        # Ignore hessian scores service as we do not use it here
        tg, bit_widths_config, _, _ = core_runner(in_model=in_model,
                                                  representative_data_gen=representative_data_gen,
                                                  core_config=core_config,
                                                  fw_info=DEFAULT_PYTORCH_INFO,
                                                  fw_impl=fw_impl,
                                                  fqc=framework_platform_capabilities,
                                                  target_resource_utilization=target_resource_utilization,
                                                  tb_w=tb_w)

        tg = ptq_runner(tg, representative_data_gen, core_config, DEFAULT_PYTORCH_INFO, fw_impl, tb_w)

        _qat_wrapper = partial(qat_wrapper, qat_config=qat_config)

        qat_model, user_info = PyTorchModelBuilder(graph=tg,
                                                   fw_info=DEFAULT_PYTORCH_INFO,
                                                   wrapper=_qat_wrapper,
                                                   get_activation_quantizer_holder_fn=partial(
                                                       get_activation_quantizer_holder,
                                                       qat_config=qat_config)).build_model()

        user_info.mixed_precision_cfg = bit_widths_config

        # Remove fw_info from graph to enable saving the pytorch model (fw_info can not be pickled)
        delattr(qat_model.graph, 'fw_info')

        return qat_model, user_info


    def pytorch_quantization_aware_training_finalize_experimental(in_model: Module):
        """
         Convert a model fine-tuned by the user to a network with QuantizeWrappers containing
         InferableQuantizers, that quantizes both the layers weights and outputs

         Args:
             in_model (Model): Pytorch model to remove QuantizeWrappers.

         Returns:
             A quantized model with QuantizeWrappers and InferableQuantizers.

         Examples:

             Import MCT:

             >>> import model_compression_toolkit as mct

             Import a Pytorch model:

             >>> from torchvision.models import mobilenet_v2
             >>> model = mobilenet_v2(pretrained=True)

             Create a random dataset generator:

             >>> import numpy as np
             >>> def repr_datagen(): yield [np.random.random((1, 224, 224, 3))]

             Create a MCT core config, containing the quantization configuration:

             >>> config = mct.core.CoreConfig()

             Pass the model, the representative dataset generator, the configuration and the target resource utilization to get a
             quantized model:

             >>> quantized_model, quantization_info = mct.qat.pytorch_quantization_aware_training_init_experimental(model, repr_datagen, core_config=config)

             Use the quantized model for fine-tuning. Finally, remove the quantizer wrappers and keep a quantize model ready for inference.

             >>> quantized_model = mct.qat.pytorch_quantization_aware_training_finalize_experimental(quantized_model)

         """
        Logger.warning(
            f"pytorch_quantization_aware_training_finalize_experimental is experimental and is subject to future changes."
            f"If you encounter an issue, please open an issue in our GitHub "
            f"project https://github.com/sony/model_optimization")

        for _, layer in in_model.named_children():
            if isinstance(layer, (PytorchQuantizationWrapper, PytorchActivationQuantizationHolder)):
                layer.convert_to_inferable_quantizers()

        return in_model


else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_quantization_aware_training_init_experimental(*args, **kwargs):
        Logger.critical('PyTorch must be installed to use pytorch_quantization_aware_training_init_experimental. '
                        "The 'torch' package is missing.")  # pragma: no cover


    def pytorch_quantization_aware_training_finalize_experimental(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'pytorch_quantization_aware_training_finalize_experimental'. "
                        "The 'torch' package is missing.")  # pragma: no cover
