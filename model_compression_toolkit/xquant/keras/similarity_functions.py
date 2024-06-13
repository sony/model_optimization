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

import numpy as np
import tensorflow as tf

from model_compression_toolkit.xquant.common.similarity_functions import SimilarityFunctions

class KerasSimilarityFunctions(SimilarityFunctions):
    """
    A class that extends SimilarityFunctions to implement similarity metrics using Keras.
    Even though the names referred to are quantized and float, it can help compare between
    tensors of any two models.
    """

    @staticmethod
    def compute_mse(f_pred: np.ndarray, q_pred: np.ndarray) -> float:
        """
        Compute the Mean Squared Error (MSE) between the float and quantized predictions.

        Args:
            f_pred (np.ndarray): Float model predictions.
            q_pred (np.ndarray): Quantized model predictions.

        Returns:
            float: The computed MSE value.
        """
        mse = tf.keras.losses.MeanSquaredError()(f_pred, q_pred)
        return float(mse.numpy())

    @staticmethod
    def compute_cs(f_pred: np.ndarray, q_pred: np.ndarray) -> float:
        """
        Compute the Cosine Similarity (CS) between the float and quantized predictions.

        Args:
            f_pred (np.ndarray): Float model predictions.
            q_pred (np.ndarray): Quantized model predictions.

        Returns:
            float: The computed CS value.
        """
        cs = tf.keras.losses.CosineSimilarity()(f_pred.flatten(), q_pred.flatten())
        return float(cs.numpy())

    @staticmethod
    def compute_sqnr(f_pred: np.ndarray, q_pred: np.ndarray) -> float:
        """
        Compute the Signal-to-Quantization-Noise Ratio (SQNR) between the float and quantized predictions.

        Args:
            f_pred (np.ndarray): Float model predictions.
            q_pred (np.ndarray): Quantized model predictions.

        Returns:
            float: The computed SQNR value.
        """
        signal_power = tf.reduce_mean(tf.square(f_pred))
        noise_power = tf.reduce_mean(tf.square(f_pred - q_pred))
        sqnr = signal_power / noise_power
        return float(sqnr.numpy())


