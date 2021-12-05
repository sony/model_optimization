# Copyright 2021 Sony Semiconductors Israel, Inc. All rights reserved.
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

from typing import Callable, Any


class GradientPTQConfig:
    """
    Configuration to use for quantization with GradientPTQ (experimental).
    """

    def __init__(self,
                 n_iter: int,
                 optimizer: Any,
                 loss: Callable = None,
                 log_function: Callable = None,
                 train_bias: bool = True,
                 representative_data_gen: Callable = None):
        """
        Initialize a GradientPTQConfig.

        Args:
            n_iter (int): Number of iterations to train.
            optimizer (OptimizerV2): Optimizer to use.
            loss (Callable): the loss to use. should accept 2 lists of tensors. 1st list of quantized tensors, the 2nd list is the float tensors.
            log_function (Callable): Function to log information about the GPTQ process.
            train_bias (bool): Whether to update the bias during the training or not.
            representative_data_gen (Callable): Dataset generator.

        Examples:
            Create a GradientPTQConfig to run for 5 iteration and uses a random dataset generator:

            >>> import numpy as np
            >>> def repr_datagen(): return [np.random.random((1,224,224,3))]
            >>> gptq_conf = GradientPTQConfig(n_iter=5, representative_data_gen=repr_datagen)

            An optimizer can be passed:

            >>> gptq_conf = GradientPTQConfig(n_iter=5, representative_data_gen=repr_datagen, optimizer=tf.keras.optimizers.Nadam(learning_rate=0.2))

            To disable the biases training, one may set train_bias to False (enabled by default):

            >>> gptq_conf = GradientPTQConfig(n_iter=5, representative_data_gen=repr_datagen, train_bias=False)

            The configuration can then be passed to :func:`~model_compression_toolkit.keras_post_training_quantization`.

        """
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.loss = loss
        self.log_function = log_function
        self.train_bias = train_bias
        self.representative_data_gen = representative_data_gen