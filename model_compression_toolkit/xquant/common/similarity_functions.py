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

from typing import Any, Dict, Callable

from model_compression_toolkit.xquant.common.constants import CS_SIMILARITY_METRIC_NAME, SQNR_SIMILARITY_METRIC_NAME, MSE_SIMILARITY_METRIC_NAME

DEFAULT_SIMILARITY_METRICS_NAMES = [CS_SIMILARITY_METRIC_NAME, MSE_SIMILARITY_METRIC_NAME, SQNR_SIMILARITY_METRIC_NAME]

class SimilarityFunctions:
    """
    A class that provides various static methods to compute similarity metrics between tensors.
    """

    @staticmethod
    def compute_mse(x: Any, y: Any) -> float:
        """
        Compute the Mean Squared Error (MSE) between two tensors (usually, the float and quantized predictions).

        Args:
            x (Any): First tensor to compare.
            y (Any): Second tensor to compare.

        Returns:
            float: The computed MSE value.
        """
        raise NotImplemented  # pragma: no cover

    @staticmethod
    def compute_cs(x: Any, y: Any) -> float:
        """
        Compute the Cosine Similarity (CS) between two tensors (usually, the float and quantized predictions).

        Args:
            x (Any): First tensor to compare.
            y (Any): Second tensor to compare.

        Returns:
            float: The computed CS value.
        """
        raise NotImplemented  # pragma: no cover

    @staticmethod
    def compute_sqnr(x: Any, y: Any) -> float:
        """
        Compute the Signal-to-Quantization-Noise Ratio (SQNR) between two tensors (usually, the float and quantized predictions).

        Args:
            x (Any): First tensor to compare.
            y (Any): Second tensor to compare.

        Returns:
            float: The computed SQNR value.
        """
        raise NotImplemented  # pragma: no cover

    def get_default_similarity_metrics(self) -> Dict[str, Callable]:
        """
        Get the default similarity metrics to compute.

        Returns:
            Dict[str, Callable]: A dictionary where the keys are similarity metric names and the values are the corresponding functions.
        """
        return {
            MSE_SIMILARITY_METRIC_NAME: self.compute_mse,
            CS_SIMILARITY_METRIC_NAME: self.compute_cs,
            SQNR_SIMILARITY_METRIC_NAME: self.compute_sqnr
        }

