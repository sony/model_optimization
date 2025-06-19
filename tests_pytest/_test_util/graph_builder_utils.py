# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
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
from typing import Union, Iterable, List, Callable

from mct_quantizers import QuantizationMethod
from model_compression_toolkit.core import QuantizationConfig

from model_compression_toolkit.core.common import BaseNode
from model_compression_toolkit.core.common.quantization.candidate_node_quantization_config import \
    CandidateNodeQuantizationConfig
from model_compression_toolkit.core.common.quantization.node_quantization_config import \
    NodeActivationQuantizationConfig, NodeWeightsQuantizationConfig
from model_compression_toolkit.target_platform_capabilities import AttributeQuantizationConfig, OpQuantizationConfig, \
    Signedness


class DummyLayer:
    """ Only needed for repr(node) to work. """
    pass


def build_node(name='node', canonical_weights: dict = None, final_weights: dict = None,
               qcs: List[CandidateNodeQuantizationConfig] = None,
               input_shape=(4, 5, 6), output_shape=(4, 5, 6),
               layer_class=DummyLayer, reuse=False):
    """ Build a node for tests.
        Either 'canonical_weights' (to be used by default) or 'final_weights' should be passed.
          Canonical weights are converted into full unique names, that contain the canonical name as a substring.
          Final weights are used as is.
        candidate_quantization_cfg is set is qcs is passed."""
    assert canonical_weights is None or final_weights is None
    if canonical_weights:
        weights = {k if isinstance(k, int) else full_attr_name(k): w for k, w in canonical_weights.items()}
    else:
        weights = final_weights or {}
    node = BaseNode(name=name,
                    framework_attr={},
                    input_shape=input_shape,
                    output_shape=output_shape,
                    weights=weights,
                    layer_class=layer_class,
                    reuse=reuse)
    if qcs:
        assert isinstance(qcs, list)
        node.candidates_quantization_cfg = qcs
    return node


def full_attr_name(canonical_name: Union[str, dict, Iterable]):
    """ Convert canonical attr (such as 'kernel') into a full name originated from the layer (e.g. 'conv2d_1/kernel:0')
        We just need the names to differ from canonical to make sure we call the correct apis. We use the same
        template for simplicity, so we don't have to explicitly synchronize names between node and weight configs."""
    convert = lambda name: f'{name[0]}/{name}/{name[-1]}' if isinstance(name, str) else name
    if isinstance(canonical_name, str):
        return convert(canonical_name)
    assert isinstance(canonical_name, (list, tuple, set))
    return canonical_name.__class__([convert(name) for name in canonical_name])


def build_nbits_qc(a_nbits=8, a_enable=True, w_attr=None, pos_attr=(32, False, ()),
                   convert_canonical_attr=True, q_preserving=False,
                   activation_quantization_fn: Callable=None) -> CandidateNodeQuantizationConfig:
    """
    Build quantization config with configurable nbits and enabling/disabling quantization only.

    Args:
        a_nbits: activation num bits.
        a_enable: whether to enable activation quantization.
        w_attr: quantization configuration for weight attributes in format {canonical name: (nbits, q_enabled)}.
          By default, a canonical weight name is expected and is automatically converted to a dummy full name (that
          contains the canonical name as a substring).
          Final name can be passed along with convert_canonical_attr=False.
        pos_attr: quantization configuration for positional weights in format (nbits, q enabled, indices).
        convert_canonical_attr: whether to convert w_attr keys to full names.
        q_preserving: Whether node is quantization preserving.

    Returns:

    """
    assert not(a_enable and q_preserving)

    w_attr = w_attr or {}
    attr_weights_configs_mapping = {
        k: AttributeQuantizationConfig(weights_n_bits=v[0], enable_weights_quantization=v[1])
        for k, v in w_attr.items()
    }
    qc = QuantizationConfig()
    # positional attrs are set via default weight config (so all pos attrs have the same q config)
    op_cfg = OpQuantizationConfig(
        # canonical names (as 'kernel')
        attr_weights_configs_mapping=attr_weights_configs_mapping,
        activation_n_bits=a_nbits,
        enable_activation_quantization=a_enable,
        default_weight_attr_config=AttributeQuantizationConfig(weights_n_bits=pos_attr[0],
                                                               enable_weights_quantization=pos_attr[1]),
        activation_quantization_method=QuantizationMethod.POWER_OF_TWO,
        quantization_preserving=q_preserving,
        supported_input_activation_n_bits=[2, 4, 8],
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=None,
        signedness=Signedness.AUTO
    )
    a_qcfg = NodeActivationQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                              activation_quantization_fn=activation_quantization_fn,
                                              activation_quantization_params_fn=None)
    # full names from the layers
    attr_names = list(w_attr.keys())
    if convert_canonical_attr:
        attr_names = [full_attr_name(k) for k in w_attr.keys()]
    w_qcfg = NodeWeightsQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                           weights_channels_axis=None,
                                           node_attrs_list=attr_names + list(pos_attr[2]))
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg,
                                         weights_quantization_cfg=w_qcfg)

    # we generate q configs via constructors to follow the real code as closely as reasonably possible.
    # verify that we actually got the configurations we want
    assert qc.activation_quantization_cfg.activation_n_bits == a_nbits
    assert qc.activation_quantization_cfg.enable_activation_quantization is a_enable
    for k, v in w_attr.items():
        # get_attr_config accepts canonical attr names
        assert qc.weights_quantization_cfg.get_attr_config(k).weights_n_bits == v[0]
        assert qc.weights_quantization_cfg.get_attr_config(k).enable_weights_quantization == v[1]
    for pos in pos_attr[2]:
        assert qc.weights_quantization_cfg.get_attr_config(pos).weights_n_bits == pos_attr[0]
        assert qc.weights_quantization_cfg.get_attr_config(pos).enable_weights_quantization == pos_attr[1]

    return qc

def build_nbits_qc_with_quantization_method(a_nbits=8, a_enable=True, w_attr=None, pos_attr=(32, False, ()),
                   convert_canonical_attr=True, q_preserving=False,
                   activation_quantization_fn: Callable=None,
                   activation_quantization_method: QuantizationMethod=QuantizationMethod.POWER_OF_TWO) -> CandidateNodeQuantizationConfig:
    """
    This function is the same as the build_nbits_qc function above, but with the addition of activation_quantization_method.
    """
    assert not(a_enable and q_preserving)

    w_attr = w_attr or {}
    attr_weights_configs_mapping = {
        k: AttributeQuantizationConfig(weights_n_bits=v[0], enable_weights_quantization=v[1])
        for k, v in w_attr.items()
    }
    qc = QuantizationConfig()
    # positional attrs are set via default weight config (so all pos attrs have the same q config)
    op_cfg = OpQuantizationConfig(
        # canonical names (as 'kernel')
        attr_weights_configs_mapping=attr_weights_configs_mapping,
        activation_n_bits=a_nbits,
        enable_activation_quantization=a_enable,
        default_weight_attr_config=AttributeQuantizationConfig(weights_n_bits=pos_attr[0],
                                                               enable_weights_quantization=pos_attr[1]),
        activation_quantization_method=activation_quantization_method,
        quantization_preserving=q_preserving,
        supported_input_activation_n_bits=[2, 4, 8],
        fixed_scale=None,
        fixed_zero_point=None,
        simd_size=None,
        signedness=Signedness.AUTO
    )
    a_qcfg = NodeActivationQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                              activation_quantization_fn=activation_quantization_fn,
                                              activation_quantization_params_fn=None)
    # full names from the layers
    attr_names = list(w_attr.keys())
    if convert_canonical_attr:
        attr_names = [full_attr_name(k) for k in w_attr.keys()]
    w_qcfg = NodeWeightsQuantizationConfig(qc=qc, op_cfg=op_cfg,
                                           weights_channels_axis=None,
                                           node_attrs_list=attr_names + list(pos_attr[2]))
    qc = CandidateNodeQuantizationConfig(activation_quantization_cfg=a_qcfg,
                                         weights_quantization_cfg=w_qcfg)

    # we generate q configs via constructors to follow the real code as closely as reasonably possible.
    # verify that we actually got the configurations we want
    assert qc.activation_quantization_cfg.activation_n_bits == a_nbits
    assert qc.activation_quantization_cfg.enable_activation_quantization is a_enable
    for k, v in w_attr.items():
        # get_attr_config accepts canonical attr names
        assert qc.weights_quantization_cfg.get_attr_config(k).weights_n_bits == v[0]
        assert qc.weights_quantization_cfg.get_attr_config(k).enable_weights_quantization == v[1]
    for pos in pos_attr[2]:
        assert qc.weights_quantization_cfg.get_attr_config(pos).weights_n_bits == pos_attr[0]
        assert qc.weights_quantization_cfg.get_attr_config(pos).enable_weights_quantization == pos_attr[1]

    return qc
