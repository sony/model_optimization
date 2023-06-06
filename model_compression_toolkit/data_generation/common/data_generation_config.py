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
from enum import Enum
from typing import Callable, Any, Dict, List
from model_compression_toolkit.core.common.defaultdict import DefaultDict
from model_compression_toolkit.core import common


class ImageGranularity(Enum):
    """
    An enum for choosing the image dependence granularity when generating images.
    0. ImageWise
    1. BatchWise
    2. AllImages
    """
    ImageWise = 0
    BatchWise = 1
    AllImages = 2


class BaseImagePipeline:
    def __init__(self,output_image_size):
        self.output_image_size = output_image_size

    def get_image_input_size(self):
        raise NotImplemented

    def image_input_manipulation(self, images):
        raise NotImplemented

    def image_output_finalize(self, images):
        raise NotImplemented


class DataGenerationConfig:
    """
    Configuration for data generation.
    """

    def __init__(self,
                 n_iter: int,
                 optimizer: Any,
                 scheduler: Any,
                 data_gen_batch_size: int,
                 initial_lr: float,
                 image_granularity: ImageGranularity = ImageGranularity.AllImages,
                 bna_loss_fn: Callable = None,
                 image_pipeline: BaseImagePipeline = None,
                 image_initialization_fn: Callable = None,
                 bn_layer_types: List = []
                 ):
        """
        Initialize a DataGenerationConfig.

        Args:
            n_iter (int): Number of iterations to train.
            optimizer (Any): Optimizer to use.
            scheduler (Any): Scheduler to use.
            loss (Callable): The loss to use. should accept 6 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors,
             the 3rd is a list of quantized weights, the 4th is a list of float weights, the 5th and 6th lists are the mean and std of the tensors
             accordingly. see example in multiple_tensors_mse_loss
            log_function (Callable): Function to log information about the GPTQ process.
            train_bias (bool): Whether to update the bias during the training or not.
            rounding_type (RoundingType): An enum that defines the rounding type.
            use_hessian_based_weights (bool): Whether to use Hessian-based weights for weighted average loss.
            optimizer_quantization_parameter (Any): Optimizer to override the rest optimizer  for quantizer parameters.
            optimizer_bias (Any): Optimizer to override the rest optimizer for bias.
            regularization_factor (float): A floating point number that defines the regularization factor.
            hessian_weights_config (GPTQHessianWeightsConfig): A configuration that include all necessary arguments to run a computation of Hessian weights for the GPTQ loss.
            gptq_quantizer_params_override (dict): A dictionary of parameters to override in GPTQ quantizer instantiation. Defaults to None (no parameters).

        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_gen_batch_size = data_gen_batch_size
        self.initial_lr = initial_lr
        self.image_granularity = image_granularity
        self.bna_loss_fn = bna_loss_fn
        self.image_pipeline = image_pipeline
        self.image_initialization_fn = image_initialization_fn
        self.bn_layer_types = bn_layer_types

    def get_dimensions_for_average(self):
        if self.image_granularity == ImageGranularity.ImageWise:
            return [2, 3]
        else:
            return [0, 2, 3]


