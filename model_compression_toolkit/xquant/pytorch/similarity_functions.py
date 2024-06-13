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


from model_compression_toolkit.xquant.common.similarity_functions import SimilarityFunctions
import torch

class PytorchSimilarityFunctions(SimilarityFunctions):

    @staticmethod
    def compute_mse(x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes Mean Squared Error between between two tensors (usually, the float and quantized tensors).

        Args:
            x: Float model predictions.
            y: Quantized model predictions.

        Returns:
            Mean Squared Error as a float.
        """
        mse = torch.nn.functional.mse_loss(x, y)
        return mse.item()

    @staticmethod
    def compute_cs(x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes Cosine Similarity between two tensors (usually, the float and quantized tensors).

        Args:
            x: Float model predictions.
            y: Quantized model predictions.

        Returns:
            Cosine Similarity as a float.
        """
        cs = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        return cs.item()

    @staticmethod
    def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes Signal-to-Quantization-Noise Ratio between two tensors (usually, the float and quantized tensors).

        Args:
            x: Float model predictions.
            y: Quantized model predictions.

        Returns:
            Signal-to-Quantization-Noise Ratio as a float.
        """
        signal_power = torch.mean(x ** 2)
        noise_power = torch.mean((x - y) ** 2)
        sqnr = signal_power / noise_power
        return sqnr.item()

