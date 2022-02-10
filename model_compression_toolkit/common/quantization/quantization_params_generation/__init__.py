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

from model_compression_toolkit.common.quantization.quantization_params_generation.kl_selection import kl_selection_tensor, kl_selection_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.lp_selection import lp_selection_histogram, \
    lp_selection_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation.kmeans_params import kmeans_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation.lut_kmeans_params import lut_kmeans_tensor
from model_compression_toolkit.common.quantization.quantization_params_generation.mae_selection import mae_selection_tensor, mae_selection_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.mse_selection import mse_selection_tensor, mse_selection_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.no_clipping import no_clipping_selection_tensor, no_clipping_selection_min_max, no_clipping_selection_histogram
from model_compression_toolkit.common.quantization.quantization_params_generation.symmetric_selection import symmetric_no_clipping_selection_min_max
from model_compression_toolkit.common.quantization.quantization_params_generation.uniform_selection import uniform_no_clipping_selection_min_max
from model_compression_toolkit.common.quantization.quantization_params_generation.outlier_filter import z_score_filter
from model_compression_toolkit.common.quantization.quantization_params_generation.qparams_search import \
    qparams_selection_tensor_search, qparams_selection_histogram_search, quantize_tensor
