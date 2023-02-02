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
from typing import Callable
from functools import partial

from model_compression_toolkit.core.common.constants import FOUND_TORCH, PYTORCH

from model_compression_toolkit import CoreConfig
from model_compression_toolkit.core import common
from model_compression_toolkit.core.common import Logger
from model_compression_toolkit.core.common.framework_info import FrameworkInfo
from model_compression_toolkit.core.common.mixed_precision.kpi_tools.kpi import KPI
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfigV2
from model_compression_toolkit.core.common.target_platform.targetplatform2framework import TargetPlatformCapabilities
from model_compression_toolkit.core.runner import core_runner, _init_tensorboard_writer
from model_compression_toolkit.ptq.runner import ptq_runner


if FOUND_TORCH:
    import torch.nn as nn
    from torch.nn import Module
    from model_compression_toolkit.core.pytorch.default_framework_info import DEFAULT_PYTORCH_INFO
    from model_compression_toolkit.core.pytorch.constants import DEFAULT_TP_MODEL
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.qat.common.qat_config import _is_qat_applicable
    from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PyTorchModelBuilder
    from model_compression_toolkit.quantizers_infrastructure import PytorchQuantizationWrapper
    from model_compression_toolkit import quantizers_infrastructure as qi
    from model_compression_toolkit import get_target_platform_capabilities
    from model_compression_toolkit.qat.common.qat_config import QATConfig
    from model_compression_toolkit.qat.pytorch.quantizer.quantization_builder import quantization_builder
    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)


    def qat_wrapper(n: common.BaseNode, module: nn.Module, qat_config: QATConfig):
        """
        A function which takes a computational graph node and a pytorch module and perform the quantization wrapping
        Args:
            n: A node of mct graph.
            module: A Pytorch module
            qat_config (QATConfig): QAT configuration
        Returns: Wrapped layer

        """
        if _is_qat_applicable(n, DEFAULT_PYTORCH_INFO):
            weights_quantizers, activation_quantizers = quantization_builder(n, qat_config, DEFAULT_PYTORCH_INFO)
            return qi.PytorchQuantizationWrapper(module, weights_quantizers, activation_quantizers)
        else:
            return module


    def pytorch_quantization_aware_training_init(in_model: Module,
                                                 representative_data_gen: Callable,
                                                 target_kpi: KPI = None,
                                                 core_config: CoreConfig = CoreConfig(),
                                                 qat_config: QATConfig = QATConfig(),
                                                 fw_info: FrameworkInfo = DEFAULT_PYTORCH_INFO,
                                                 target_platform_capabilities: TargetPlatformCapabilities = DEFAULT_PYTORCH_TPC):
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
         In order to limit the maximal model's size, a target KPI need to be passed after weights_memory
         is set (in bytes).

         Args:
             in_model (Model): Pytorch model to quantize.
             representative_data_gen (Callable): Dataset used for initial calibration.
             target_kpi (KPI): KPI object to limit the search of the mixed-precision configuration as desired.
             core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
             qat_config (QATConfig): QAT configuration
             fw_info (FrameworkInfo): Information needed for quantization about the specific framework (e.g., kernel channels indices, groups of layers by how they should be quantized, etc.).  `Default Pytorch info <https://github.com/sony/model_optimization/blob/main/model_compression_toolkit/core/pytorch/default_framework_info.py>`_
             target_platform_capabilities (TargetPlatformCapabilities): TargetPlatformCapabilities to optimize the Pytorch model according to.

         Returns:

             A quantized model.
             User information that may be needed to handle the quantized model.

         Examples:

             Import MCT:

             >>> import model_compression_toolkit as mct

             Import a Pytorch model:

             >>> from torchvision.models import mobilenet_v2
             >>> model = mobilenet_v2(pretrained=True)

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 3, 224, 224))]

             Create a MCT core config, containing the quantization configuration:

             >>> config = mct.CoreConfig()

             Pass the model, the representative dataset generator, the configuration and the target KPI to get a
             quantized model. Now the model contains quantizer wrappers for fine tunning the weights:

             >>> quantized_model, quantization_info = pytorch_quantization_aware_training_init(model, repr_datagen, core_config=config)

             For more configuration options, please take a look at our `API documentation <https://sony.github.io/model_optimization/api/api_docs/modules/mixed_precision_quantization_config.html>`_.

         """

        if core_config.mixed_precision_enable:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfigV2):
                common.Logger.error("Given quantization config to mixed-precision facade is not of type "
                                    "MixedPrecisionQuantizationConfigV2. Please use pytorch_post_training_quantization API,"
                                    "or pass a valid mixed precision configuration.")

            common.Logger.info("Using experimental mixed-precision quantization. "
                               "If you encounter an issue please file a bug.")

        tb_w = _init_tensorboard_writer(fw_info)

        fw_impl = PytorchImplementation()

        tg, bit_widths_config = core_runner(in_model=in_model,
                                            representative_data_gen=representative_data_gen,
                                            core_config=core_config,
                                            fw_info=DEFAULT_PYTORCH_INFO,
                                            fw_impl=fw_impl,
                                            tpc=target_platform_capabilities,
                                            target_kpi=target_kpi,
                                            tb_w=tb_w)

        tg = ptq_runner(tg, representative_data_gen, core_config, fw_info, fw_impl, tb_w)

        _qat_wrapper = partial(qat_wrapper, qat_config=qat_config)

        qat_model, user_info = PyTorchModelBuilder(graph=tg, fw_info=fw_info, wrapper=_qat_wrapper).build_model()

        user_info.mixed_precision_cfg = bit_widths_config

        return qat_model, user_info

    def pytorch_quantization_aware_training_finalize(in_model: Module):
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

             >>> config = mct.CoreConfig()

             Pass the model, the representative dataset generator, the configuration and the target KPI to get a
             quantized model:

             >>> quantized_model, quantization_info = pytorch_quantization_aware_training_init(model, repr_datagen, core_config=config)

             Use the quantized model for fine-tuning. Finally, remove the quantizer wrappers and keep a quantize model ready for inference.

             >>> quantized_model = mct.pytorch_quantization_aware_training_finalize(quantized_model)

         """
        exported_model = copy.deepcopy(in_model)
        for _, layer in exported_model.named_children():
            if isinstance(layer, PytorchQuantizationWrapper):
                layer.convert_to_inferable_quantizers()

        return exported_model


else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_quantization_aware_training_init(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_quantization_aware_training_init. '
                        'Could not find the torch package.')  # pragma: no cover

    def pytorch_quantization_aware_training_finalize(*args, **kwargs):
        Logger.critical('Installing Pytorch is mandatory '
                        'when using pytorch_quantization_aware_training_finalize. '
                        'Could not find the torch package.')  # pragma: no cover
