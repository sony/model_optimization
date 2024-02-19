# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
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


def get_tpc_info(tpc):
    # Retrieve the number of bits used for activation functions within the TPC's default operation.
    activation_nbits = tpc.get_default_op_qc().activation_n_bits

    # Retrieve the number of bits used for weights in the TPC's default weight configuration.
    weights_nbits = tpc.get_default_op_qc().default_weight_attr_config.weights_n_bits

    # Extract the name of the tp model associated with the TPC.
    tp_model_name = tpc.tp_model.name

    # Get the version of the TPC.
    version = tpc.version

    return TPCInfo(activation_nbits=activation_nbits,
                   weights_nbits=weights_nbits,
                   tp_model_name=tp_model_name,
                   version=version)

class TPCInfo:
    def __init__(self,
                 activation_nbits: int,
                 weights_nbits: int,
                 tp_model_name: str,
                 version: str):
        """
        Args:
            activation_nbits: Number of bits used for activation functions.
            weights_nbits: Number of bits used for weights.
            tp_model_name: TP model's name.
            version: TPC's version.
        """
        self.activation_nbits = activation_nbits
        self.weights_nbits = weights_nbits
        self.tp_model_name = tp_model_name
        self.version = version


