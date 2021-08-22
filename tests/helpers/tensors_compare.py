# ===============================================================================
# Copyright (c) 2021, Sony Semiconductors Israel, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


import numpy as np


def check_power_of_two(x):
    return np.log2(np.abs(x)) == int(np.log2(np.abs(x)))


def check_quantizer_min_max_are_power_of_two(min_q, max_q, nbits=8):
    is_min_pow_of_two = check_power_of_two(min_q)
    lsb = (max_q - min_q) / (2 ** nbits - 1)
    is_max_pow_of_two = check_power_of_two(max_q + lsb)
    return is_min_pow_of_two, is_max_pow_of_two


def cosine_similarity(a, b, eps=1e-8):
    if np.all(b == 0) and np.all(a == 0):
        return 1.0
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = tensor_norm(a)
    b_norm = tensor_norm(b)

    return np.sum(a_flat * b_flat) / ((a_norm * b_norm) + eps)


def norm_similarity(a, b):
    return tensor_norm(a) / tensor_norm(b)


def normalized_mse(a, b, norm_factor):
    lsb_error = (np.abs(a - b) / norm_factor)
    return np.mean(lsb_error), np.std(lsb_error), np.max(lsb_error), np.min(lsb_error)


def tensor_norm(a):
    return np.sqrt(np.power(a.flatten(), 2.0).sum())


def tensor_compare(a, b):
    cs_value = cosine_similarity(a, b)
    norm_rate = tensor_norm(a) / tensor_norm(b)
    return cs_value, norm_rate
