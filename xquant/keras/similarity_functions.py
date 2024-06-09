#  Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

from functools import partial

import keras
import logging

from mct_quantizers.keras.quantize_wrapper import KerasQuantizationWrapper

from typing import Any, Dict, Callable, List, Tuple
import numpy as np
import tensorflow as tf

from xquant.common.similarity_metrics import SimilarityFunctions


class KerasSimilarityFunctions(SimilarityFunctions):

    @staticmethod
    def compute_mse(f_pred: np.ndarray, q_pred: np.ndarray) -> float:
        mse = tf.keras.losses.MeanSquaredError()(f_pred, q_pred)
        return float(mse.numpy())

    @staticmethod
    def compute_cs(f_pred: np.ndarray, q_pred: np.ndarray) -> float:
        cs = tf.keras.losses.CosineSimilarity()(f_pred.flatten(), q_pred.flatten())
        return float(cs.numpy())

    @staticmethod
    def compute_sqnr(f_pred: np.ndarray, q_pred: np.ndarray) -> float:
        signal_power = tf.reduce_mean(tf.square(f_pred))
        noise_power = tf.reduce_mean(tf.square(f_pred - q_pred))
        sqnr = signal_power / noise_power
        return float(sqnr.numpy())


