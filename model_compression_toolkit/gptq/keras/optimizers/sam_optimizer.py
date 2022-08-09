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
from typing import List, Callable, Tuple

import tensorflow as tf
from model_compression_toolkit.gptq.keras.quantizer.configs.weight_quantizer_gptq_config import WeightQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper


class SAM:
    """
    This class implements Sharpness-Aware Minimization for Efficiently Improving Generalization (https://arxiv.org/abs/2010.01412)
    """

    def __init__(self, model2quantized,
                 gradient_step: Callable,
                 optimizer_with_param: List[Tuple[List, List[tf.Tensor]]],
                 rho: float = 0.01,
                 eps: float = 1e-12):
        """
        The init function of Sharpness-Aware Minimization  gradient computation class.
        Args:
            model2quantized: Input quantized module
            gradient_step: A function that returns a list of gradients tensors
            optimizer_with_param: A list of optimizer classes to update with the corresponding parameters.
            rho: A floating point number that set the region of smoothness
            eps: A floating point number for numeric stability
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.eps = eps
        self.gradient_step = gradient_step

        self.trainable_variables = [p for _, p in optimizer_with_param]
        self.m_var = [len(p) for p in self.trainable_variables]
        self.n_groups = len(self.trainable_variables)
        self.model2quantized = model2quantized
        self.e_ws = [[] for _ in range(len(optimizer_with_param))]

    def _enable_update_step_param(self):
        """
        This function enables the parameter update (update iteration index and gumbel random variable)
        Returns: None

        """
        for layer in self.model2quantized.layers:
            if isinstance(layer, QuantizeWrapper) and isinstance(
                    layer.quantize_config, WeightQuantizeConfig):
                layer.quantize_config.enable_update()

    def _disable_update_step_param(self):
        """
        This function disables the parameter update (update iteration index and gumbel random variable)
        Returns: None

        """
        for layer in self.model2quantized.layers:
            if isinstance(layer, QuantizeWrapper) and isinstance(
                    layer.quantize_config, WeightQuantizeConfig):
                layer.quantize_config.disable_update()

    def _update_w_location(self, gradients: List[List[tf.Tensor]]):
        """
        This function updates the weights position to the highest point
        Args:
            gradients: A list of gradients tensors

        Returns: None

        """

        for g in range(self.n_groups):
            self.e_ws[g].clear()
            grad_norm = tf.linalg.global_norm(gradients[g])
            ew_multiplier = self.rho / (grad_norm + self.eps)
            for i in range(self.m_var[g]):
                e_w = tf.math.multiply(gradients[g][i], ew_multiplier)
                self.trainable_variables[g][i].assign_add(e_w)
                self.e_ws[g].append(e_w)

    def _restore_w_location(self):
        """
        Restore weights to the original position
        Returns: None

        """
        for g in range(self.n_groups):
            for i in range(self.m_var[g]):
                self.trainable_variables[g][i].assign_add(-self.e_ws[g][i])

    def compute_gradients(self, *arg, **kwargs) -> (tf.Tensor, List[List[tf.Tensor]]):
        """
        This function compute the gradients of SAM optimizer
        Args:
            *arg: args to pass to the gradient step functions
            **kwargs: kwargs to pass to the gradient step functions

        Returns: A tensor of the loss value and a list of gradients tensors

        """
        self._enable_update_step_param()
        loss, grad = self.gradient_step(*arg, **kwargs)
        self._update_w_location(grad)
        self._disable_update_step_param()
        loss, grad = self.gradient_step(*arg, **kwargs)
        self._restore_w_location()
        return loss, grad
