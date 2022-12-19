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

from typing import Dict, List

from model_compression_toolkit.qunatizers_infrastructure.common.base_quantizer import BaseQuantizer

# TODO: case of non tf
from keras.utils import serialize_keras_object, deserialize_keras_object


class NodeQuantizationDispatcher(object):
    def __init__(self,
                 weight_quantizer: Dict[str, BaseQuantizer] = None,
                 activation_quantizers: List[BaseQuantizer] = None):
        self.weight_quantizer = weight_quantizer if weight_quantizer is not None else dict()
        self.activation_quantizers = activation_quantizers if activation_quantizers is not None else list()

    def add_weight_quantizer(self, param_name, quantizer):
        self.weight_quantizer.update({param_name: quantizer})

    @property
    def is_activation_quantization(self):
        return len(self.activation_quantizers) > 0

    @property
    def is_weights_quantization(self):
        return len(self.weight_quantizer) > 0

    def get_config(self) -> dict:
        return {"activation_quantizers": [serialize_keras_object(act) for act in self.activation_quantizers],
                "weight_quantizer": {k: serialize_keras_object(v) for k, v in self.weight_quantizer.items()}}

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        activation_quantizers = [deserialize_keras_object(act,
                                                          module_objects=globals(),
                                                          custom_objects=None) for act in
                                 config.get("activation_quantizers")]
        weight_quantizer = {k: deserialize_keras_object(v,
                                                        module_objects=globals(),
                                                        custom_objects=None) for k, v in
                            config.get("weight_quantizer").items()}
        return cls(weight_quantizer, activation_quantizers)
