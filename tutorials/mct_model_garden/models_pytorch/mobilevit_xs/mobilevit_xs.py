# ---------------------------------------------------------------
#    Copyright 2019 Ross Wightman
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
# ---------------------------------------------------------------

"""
MobileViT (extra small-sized model) - PyTorch implementation

This code contains a PyTorch implementation of mobilevit-xs model, following
https://github.com/huggingface/pytorch-image-models. This implementation includes a slightly modified version of
MobileViT attention that was optimized for model quantization.

Main changes:
  - Adjust the Attention layer to enhance compatibility for quantization (renamed as ModifiedAttention).
  - Rearrange the input structure for every Attention layer to make it suitable for quantization.
  - Inheritance class from HuggingFace
  - Simplification of model initialization procedures.

The code is organized as follows:
  - Helper functions of timm building blocks, including: get_act_layer, _create_act, _create_fc, get_norm_act_layer,
    create_conv2d_pad, create_conv2d, update_block_kwargs, create_block, create_byob_stages, create_byob_stem,
    create_classifier and more.
  - Configurations of MobileViT-XS model and building blocks: ByoModelCfg, ByoBlockCfg, _inverted_residual_block,
    _mobilevit_block and model_cfgs.
  - Classes definitions of MobileViT-XS building blocks: BatchNormAct2d, ConvNormAct, BottleneckBlock,
    Attention (ModifiedAttention), Mlp, TransformerBlock, MobileVitBlock, SelectAdaptivePool2d and ClassifierHead.
  - Classification Model definition: MobileViTXSPyTorch

For more details on the mobilevit-xs model, refer to the original repository:
https://github.com/huggingface/pytorch-image-models

"""
import collections.abc
import math
import types
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial
from itertools import repeat
from typing import Tuple, Union, Optional, Any, Callable, Dict, Type, List, Sequence

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import _assert
from torch import nn
from torch.nn import functional as F

_ACT_LAYER_DEFAULT = dict(
    silu=nn.SiLU,
    swish=nn.SiLU,
    mish=nn.Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    leaky_relu=nn.LeakyReLU,
    elu=nn.ELU,
    celu=nn.CELU,
    selu=nn.SELU,
    hard_sigmoid=nn.Hardsigmoid,
    hard_swish=nn.Hardswish,
    identity=nn.Identity,
)


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


FormatT = Union[str, Format]


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
_int_tuple_2_t = Union[int, Tuple[int, int]]


def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def get_act_layer(name: Union[Type[nn.Module], str] = 'relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name is None:
        return None
    if not isinstance(name, str):
        return name
    if not name:
        return None
    return _ACT_LAYER_DEFAULT[name]


def _create_act(act_layer, act_kwargs=None, inplace=False, apply_act=True):
    act_layer = get_act_layer(act_layer)  # string -> nn.Module
    act_kwargs = act_kwargs or {}
    if act_layer is not None and apply_act:
        if inplace:
            act_kwargs['inplace'] = inplace
        act = act_layer(**act_kwargs)
    else:
        act = nn.Identity()
    return act


def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = nn.Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = nn.Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = nn.Linear(num_features, num_classes, bias=True)
    return fc


def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1


def _init_weights(module, name='', zero_init_last=False):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights(zero_init_last=zero_init_last)


@dataclass
class ByoBlockCfg:
    type: Union[str, nn.Module]
    d: int  # block depth (number of block repeats in stage)
    c: int  # number of output channels for each block in stage
    s: int = 2  # stride of stage (first block)
    gs: Optional[Union[int, Callable]] = None  # group-size of blocks in stage, conv is depthwise if gs == 1
    br: float = 1.  # bottleneck-ratio of blocks in stage

    # NOTE: these config items override the model cfgs that are applied to all blocks by default
    attn_layer: Optional[str] = None
    attn_kwargs: Optional[Dict[str, Any]] = None
    self_attn_layer: Optional[str] = None
    self_attn_kwargs: Optional[Dict[str, Any]] = None
    block_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class ByoModelCfg:
    blocks: Tuple[Union[ByoBlockCfg, Tuple[ByoBlockCfg, ...]], ...]
    downsample: str = 'conv1x1'
    stem_type: str = '3x3'
    stem_pool: Optional[str] = 'maxpool'
    stem_chs: int = 32
    width_factor: float = 1.0
    num_features: int = 0  # num out_channels for final conv, no final 1x1 conv if 0
    zero_init_last: bool = True  # zero init last weight (usually bn) in residual path
    fixed_input_size: bool = False  # model constrained to a fixed-input size / img_size must be provided on creation

    act_layer: str = 'relu'
    norm_layer: str = 'batchnorm'

    # NOTE: these config items will be overridden by the block cfg (per-block) if they are set there
    attn_layer: Optional[str] = None
    attn_kwargs: dict = field(default_factory=lambda: dict())
    self_attn_layer: Optional[str] = None
    self_attn_kwargs: dict = field(default_factory=lambda: dict())
    block_kwargs: Dict[str, Any] = field(default_factory=lambda: dict())


def _inverted_residual_block(d, c, s, br=4.0):
    # inverted residual is a bottleneck block with bottle_ratio > 1 applied to in_chs, linear output, gs=1 (depthwise)
    return ByoBlockCfg(
        type='bottle', d=d, c=c, s=s, gs=1, br=br,
        block_kwargs=dict(bottle_in=True, linear_out=True))


def _mobilevit_block(d, c, s, transformer_dim, transformer_depth, patch_size=4, br=4.0):
    # inverted residual + mobilevit blocks as per MobileViT network
    return (
        _inverted_residual_block(d=d, c=c, s=s, br=br),
        ByoBlockCfg(
            type='mobilevit', d=1, c=c, s=1,
            block_kwargs=dict(
                transformer_dim=transformer_dim,
                transformer_depth=transformer_depth,
                patch_size=patch_size)
        )
    )


model_cfgs = dict(
    mobilevit_xs=ByoModelCfg(
        blocks=(
            _inverted_residual_block(d=1, c=32, s=1),
            _inverted_residual_block(d=3, c=48, s=2),
            _mobilevit_block(d=1, c=64, s=2, transformer_dim=96, transformer_depth=2, patch_size=2),
            _mobilevit_block(d=1, c=80, s=2, transformer_dim=120, transformer_depth=4, patch_size=2),
            _mobilevit_block(d=1, c=96, s=2, transformer_dim=144, transformer_depth=3, patch_size=2),
        ),
        stem_chs=16,
        stem_type='3x3',
        stem_pool='',
        downsample='',
        act_layer='silu',
        num_features=384,
    ),
)


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=nn.ReLU,
            act_kwargs=None,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None,
    ):
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.act = _create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        _assert(x.ndim == 4, f'expected 4D input (got {x.ndim}D input)')

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = F.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


_NORM_ACT_MAP = dict(
    batchnorm=BatchNormAct2d
)
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}

# has act_layer arg to define act type
_NORM_ACT_REQUIRES_ARG = {
    BatchNormAct2d}


def get_norm_act_layer(norm_layer, act_layer=None):
    if norm_layer is None:
        return None
    assert isinstance(norm_layer, (type, str, types.FunctionType, partial))
    assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, partial))
    norm_act_kwargs = {}

    if isinstance(norm_layer, str):
        if not norm_layer:
            return None
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP[layer_name]

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


# Can SAME padding for given args be done statically?
def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, _ = get_padding_value(padding, kernel_size, **kwargs)
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    # Here, we've removed the options for returning Conv2dSame, MixedConv2d, or CondConv2d, as they aren't relevant
    # to the mobilevit-xs model.
    depthwise = kwargs.pop('depthwise', False)
    # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
    groups = in_channels if depthwise else kwargs.pop('groups', 1)
    m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


class ConvNormAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding='',
            dilation=1,
            groups=1,
            bias=False,
            apply_act=True,
            norm_layer=nn.BatchNorm2d,
            norm_kwargs=None,
            act_layer=nn.ReLU,
            act_kwargs=None,
            drop_layer=None,
    ):
        super(ConvNormAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}

        self.conv = create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        # NOTE for backwards compatibility with models that use separate norm and act layer definitions
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
        if drop_layer:
            norm_kwargs['drop_layer'] = drop_layer
        self.bn = norm_act_layer(
            out_channels,
            apply_act=apply_act,
            act_kwargs=act_kwargs,
            **norm_kwargs,
        )

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


@dataclass
class LayerFn:
    conv_norm_act: Callable = ConvNormAct
    norm_act: Callable = BatchNormAct2d
    act: Callable = nn.ReLU
    attn: Optional[Callable] = None
    self_attn: Optional[Callable] = None


def override_kwargs(block_kwargs, model_kwargs):
    """ Override model level attn/self-attn/block kwargs w/ block level

    NOTE: kwargs are NOT merged across levels, block_kwargs will fully replace model_kwargs
    for the block if set to anything that isn't None.

    i.e. an empty block_kwargs dict will remove kwargs set at model level for that block
    """
    out_kwargs = block_kwargs if block_kwargs is not None else model_kwargs
    return out_kwargs or {}  # make sure None isn't returned


def update_block_kwargs(block_kwargs: Dict[str, Any], block_cfg: ByoBlockCfg, model_cfg: ByoModelCfg, ):
    layer_fns = block_kwargs['layers']
    block_kwargs['layers'] = layer_fns

    # add additional block_kwargs specified in block_cfg or model_cfg, precedence to block if set
    block_kwargs.update(override_kwargs(block_cfg.block_kwargs, model_cfg.block_kwargs))


def expand_blocks_cfg(stage_blocks_cfg: Union[ByoBlockCfg, Sequence[ByoBlockCfg]]) -> List[ByoBlockCfg]:
    if not isinstance(stage_blocks_cfg, Sequence):
        stage_blocks_cfg = (stage_blocks_cfg,)
    block_cfgs = []
    for i, cfg in enumerate(stage_blocks_cfg):
        block_cfgs += [replace(cfg, d=1) for _ in range(cfg.d)]
    return block_cfgs


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def create_shortcut(
        downsample_type: str,
        in_chs: int,
        out_chs: int,
        stride: int,
        dilation: Tuple[int, int],
        layers: LayerFn,
        **kwargs,
):
    assert downsample_type in ('avg', 'conv1x1', '')
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        if not downsample_type:
            return None  # no shortcut
        return layers.conv_norm_act(in_chs, out_chs, kernel_size=1, stride=stride, dilation=dilation[0], **kwargs)
    else:
        return nn.Identity()  # identity shortcut


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class BottleneckBlock(nn.Module):
    """ ResNet-like Bottleneck Block - 1x1 - kxk - 1x1
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int = 3,
            stride: int = 1,
            dilation: Tuple[int, int] = (1, 1),
            bottle_ratio: float = 1.,
            group_size: Optional[int] = None,
            downsample: str = 'avg',
            attn_last: bool = False,
            linear_out: bool = False,
            extra_conv: bool = False,
            bottle_in: bool = False,
            layers: LayerFn = None,
            drop_block: Callable = None,
            drop_path_rate: float = 0.,
    ):
        super(BottleneckBlock, self).__init__()
        layers = layers or LayerFn()
        mid_chs = make_divisible((in_chs if bottle_in else out_chs) * bottle_ratio)
        groups = num_groups(group_size, mid_chs)

        self.shortcut = create_shortcut(
            downsample, in_chs, out_chs,
            stride=stride, dilation=dilation, apply_act=False, layers=layers,
        )

        self.conv1_1x1 = layers.conv_norm_act(in_chs, mid_chs, 1)
        self.conv2_kxk = layers.conv_norm_act(
            mid_chs, mid_chs, kernel_size,
            stride=stride, dilation=dilation[0], groups=groups, drop_layer=drop_block,
        )
        if extra_conv:
            self.conv2b_kxk = layers.conv_norm_act(
                mid_chs, mid_chs, kernel_size, dilation=dilation[1], groups=groups)
        else:
            self.conv2b_kxk = nn.Identity()
        self.attn = nn.Identity() if attn_last or layers.attn is None else layers.attn(mid_chs)
        self.conv3_1x1 = layers.conv_norm_act(mid_chs, out_chs, 1, apply_act=False)
        self.attn_last = nn.Identity() if not attn_last or layers.attn is None else layers.attn(out_chs)
        self.drop_path = nn.Identity()
        self.act = nn.Identity() if linear_out else layers.act(inplace=True)

    def init_weights(self, zero_init_last: bool = False):
        if zero_init_last and self.shortcut is not None and getattr(self.conv3_1x1.bn, 'weight', None) is not None:
            nn.init.zeros_(self.conv3_1x1.bn.weight)
        for attn in (self.attn, self.attn_last):
            if hasattr(attn, 'reset_parameters'):
                attn.reset_parameters()

    def forward(self, x):
        shortcut = x
        x = self.conv1_1x1(x)
        x = self.conv2_kxk(x)
        x = self.conv2b_kxk(x)
        x = self.attn(x)
        x = self.conv3_1x1(x)
        x = self.attn_last(x)
        x = self.drop_path(x)
        if self.shortcut is not None:
            x = x + self.shortcut(shortcut)
        return self.act(x)


class ModifiedAttention(nn.Module):
    """
    The ModifiedAttention class is derived from the timm/Attention class.
    We've adjusted the class to prevent folding on the batch axis and to refrain from performing matmul on tensors
    with more than 3 dimensions (considering the batch axis).
    Additionally, we've included the patch_area in the initialization to address the issue of 'Proxy' object
    interpretation in torch.fx.
    Despite these modifications, the module retains its original functionality.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            patch_area: int = 4,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Here, we've opted to include the patch_area directly instead of retrieving it within the forward method.
        self.patch_area = patch_area

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, P, N, C = x.shape
        # [B, P, N, 3*C] --> [B, P, N, 3*C]
        qkv = self.qkv(x)
        # [B, P, N, 3*C] --> [B, P, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(B, P, N, 3, self.num_heads, self.head_dim)
        # [B, P, N, 3, num_heads, head_dim] --> [B, 3, num_heads, P, N, head_dim]
        qkv = qkv.permute(0, 3, 4, 1, 2, 5)
        # [B, 3, num_heads, P, N, head_dim] --> 3 * [B, num_heads, P, N, head_dim]
        q, k, v = qkv.unbind(1)

        # We've adjusted this section to calculate the attention individually for each head and patch.
        head_list = []

        # [B, num_heads, P, N, head_dim] --> num_heads * [B, P, N, head_dim]
        q_split = q.unbind(1)
        k_split = k.unbind(1)
        v_split = v.unbind(1)
        for head in range(self.num_heads):
            # [B, P, N, head_dim] --> P * [B, N, head_dim]
            k_head = k_split[head].unbind(1)
            q_head = q_split[head].unbind(1)
            v_head = v_split[head].unbind(1)

            iter_list = []
            # Calculate the attention score head and patch
            for patch in range(self.patch_area):
                # [B, N, head_dim]
                k_patch = k_head[patch]
                q_patch = q_head[patch]
                v_patch = v_head[patch]

                k_patch = self.k_norm(k_patch)
                q_patch = self.q_norm(q_patch)

                q_patch = q_patch * self.scale

                # [B, N, head_dim] --> [B, head_dim, N]
                k_patch = k_patch.transpose(-2, -1)

                # [B, N, head_dim] @ [B, head_dim, N] --> [B, N, N]
                attn_iter = q_patch @ k_patch

                attn_iter = attn_iter.softmax(dim=-1)
                attn_iter = self.attn_drop(attn_iter)

                # [B, N, N] @ [B, N, head_dim] --> [B, N, head_dim]
                x_iter = attn_iter @ v_patch

                # P * [B, N, head_dim]
                iter_list.append(x_iter)

            # P * [B, N, head_dim] --> [B, P, N, head_dim]
            output_stacked = torch.stack(iter_list, dim=1)

            # num_heads * [B, P, N, head_dim]
            head_list.append(output_stacked)

        # num_heads * [B, P, N, head_dim] --> [B, P, num_heads, N, head_dim]
        concat_heads = torch.stack(head_list, dim=2)

        # [B, P, num_heads, N, head_dim] --> [B, P, N, num_heads, head_dim]
        x = concat_heads.transpose(2, 3)

        # [B, P, N, num_heads, head_dim] --> [B, P, N, C]
        x = x.reshape(B, P, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            patch_area: float = 4.
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ModifiedAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            patch_area=int(patch_area),
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MobileVitBlock(nn.Module):
    """ MobileViT block
        Paper: https://arxiv.org/abs/2110.02178?context=cs.LG
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            bottle_ratio: float = 1.0,
            group_size: Optional[int] = None,
            dilation: Tuple[int, int] = (1, 1),
            mlp_ratio: float = 2.0,
            transformer_dim: Optional[int] = None,
            transformer_depth: int = 2,
            patch_size: int = 8,
            num_heads: int = 4,
            attn_drop: float = 0.,
            drop: int = 0.,
            no_fusion: bool = False,
            drop_path_rate: float = 0.,
            layers: LayerFn = None,
            transformer_norm_layer: Callable = nn.LayerNorm,
            **kwargs,  # eat unused args
    ):
        super(MobileVitBlock, self).__init__()

        layers = layers or LayerFn()
        groups = num_groups(group_size, in_chs)
        out_chs = out_chs or in_chs
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_chs)

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

        self.conv_kxk = layers.conv_norm_act(
            in_chs, in_chs, kernel_size=kernel_size,
            stride=stride, groups=groups, dilation=dilation[0])
        self.conv_1x1 = nn.Conv2d(in_chs, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(*[
            TransformerBlock(
                transformer_dim,
                mlp_ratio=mlp_ratio,
                num_heads=num_heads,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=drop,
                drop_path=drop_path_rate,
                act_layer=layers.act,
                norm_layer=transformer_norm_layer,
                patch_area=self.patch_area,
            )
            for _ in range(transformer_depth)
        ])
        self.norm = transformer_norm_layer(transformer_dim)

        self.conv_proj = layers.conv_norm_act(transformer_dim, out_chs, kernel_size=1, stride=1)

        if no_fusion:
            self.conv_fusion = None
        else:
            self.conv_fusion = layers.conv_norm_act(in_chs + out_chs, out_chs, kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        # Local representation
        x = self.conv_kxk(x)
        x = self.conv_1x1(x)

        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N
        interpolate = False

        # We've adjusted this part to avoid folding on the batch axis.
        # We've made a change here. Instead of fetching the shape as [B * C * n_h, n_w, p_h, p_w], we now fetch it as
        # [B, C * n_h, n_w, p_h, p_w].
        # [B, C, H, W] --> [B, C * n_h, p_h, n_w, p_w]
        x = x.reshape(B, C * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B, C * n_h, p_h, n_w, p_w] --> [B, C * n_h, n_w, p_h, p_w]
        x = x.transpose(2, 3)

        # We've made a change here. Instead of fetching the shape as [BP, N, C], we now fetch it as [B, P, N, C].
        # [B, C * n_h, n_w, p_h, p_w] --> [B, C, N, P]
        x = x.reshape(B, C, num_patches, self.patch_area)
        # [B, C, N, P]  --> [B, P, N, C]
        x = x.transpose(1, 3)

        # Global representations
        x = self.transformer(x)
        x = self.norm(x)

        # We've adjusted this part to avoid folding on the batch axis.
        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] --> [B, C * n_h, n_w, p_h, p_w]
        x = x.reshape(B, C * num_patch_h, num_patch_w, patch_h, patch_w)
        # [B, C * n_h, n_w, p_h, p_w] --> [B, C * n_h, p_h, n_w, p_w]
        x = x.transpose(2, 3)
        # [B, C * n_h, p_h, n_w, p_w] --> [B, C, n_h * p_h, n_w * p_w]
        x = x.reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        if interpolate:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

        x = self.conv_proj(x)
        if self.conv_fusion is not None:
            x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        return x


_block_registry = dict(
    bottle=BottleneckBlock,
    mobilevit=MobileVitBlock,
)


def create_block(block: Union[str, nn.Module], **kwargs):
    if isinstance(block, (nn.Module, partial)):
        return block(**kwargs)
    assert block in _block_registry, f'Unknown block type ({block}'
    return _block_registry[block](**kwargs)


def create_byob_stages(
        cfg: ByoModelCfg,
        drop_path_rate: float,
        output_stride: int,
        stem_feat: Dict[str, Any],
        feat_size: Optional[int] = None,
        layers: Optional[LayerFn] = None,
        block_kwargs_fn: Optional[Callable] = update_block_kwargs,
):
    layers = layers or LayerFn()
    feature_info = []
    block_cfgs = [expand_blocks_cfg(s) for s in cfg.blocks]
    depths = [sum([bc.d for bc in stage_bcs]) for stage_bcs in block_cfgs]
    dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
    dilation = 1
    net_stride = stem_feat['reduction']
    prev_chs = stem_feat['num_chs']
    prev_feat = stem_feat
    stages = []
    for stage_idx, stage_block_cfgs in enumerate(block_cfgs):
        stride = stage_block_cfgs[0].s
        if stride != 1 and prev_feat:
            feature_info.append(prev_feat)
        if net_stride >= output_stride and stride > 1:
            dilation *= stride
            stride = 1
        net_stride *= stride
        first_dilation = 1 if dilation in (1, 2) else 2

        blocks = []
        for block_idx, block_cfg in enumerate(stage_block_cfgs):
            out_chs = make_divisible(block_cfg.c * cfg.width_factor)
            group_size = block_cfg.gs
            if isinstance(group_size, Callable):
                group_size = group_size(out_chs, block_idx)
            block_kwargs = dict(  # Blocks used in this model must accept these arguments
                in_chs=prev_chs,
                out_chs=out_chs,
                stride=stride if block_idx == 0 else 1,
                dilation=(first_dilation, dilation),
                group_size=group_size,
                bottle_ratio=block_cfg.br,
                downsample=cfg.downsample,
                drop_path_rate=dpr[stage_idx][block_idx],
                layers=layers,
            )
            if block_cfg.type in ('self_attn',):
                # add feat_size arg for blocks that support/need it
                block_kwargs['feat_size'] = feat_size
            block_kwargs_fn(block_kwargs, block_cfg=block_cfg, model_cfg=cfg)
            blocks += [create_block(block_cfg.type, **block_kwargs)]
            first_dilation = dilation
            prev_chs = out_chs
            if stride > 1 and block_idx == 0:
                feat_size = reduce_feat_size(feat_size, stride)

        stages += [nn.Sequential(*blocks)]
        prev_feat = dict(num_chs=prev_chs, reduction=net_stride, module=f'stages.{stage_idx}')

    feature_info.append(prev_feat)
    return nn.Sequential(*stages), feature_info


def create_byob_stem(
        in_chs: int,
        out_chs: int,
        stem_type: str = '',
        pool_type: str = '',
        feat_prefix: str = 'stem',
        layers: LayerFn = None,
):
    layers = layers or LayerFn()
    stem = layers.conv_norm_act(in_chs, out_chs, 3, stride=2)
    feature_info = [dict(num_chs=out_chs, reduction=2, module=feat_prefix)]
    return stem, feature_info


def reduce_feat_size(feat_size, stride=2):
    return None if feat_size is None else tuple([s // stride for s in feat_size])


def get_layer_fns(cfg: ByoModelCfg):
    act = get_act_layer(cfg.act_layer)
    norm_act = get_norm_act_layer(norm_layer=cfg.norm_layer, act_layer=act)
    conv_norm_act = partial(ConvNormAct, norm_layer=cfg.norm_layer, act_layer=act)
    # To streamline the process, we've opted to set None for attn and self_attn instead of invoking the get_attn
    # function, in line with the configuration of the mobilevit-xs model.

    attn = None
    self_attn = None
    layer_fn = LayerFn(conv_norm_act=conv_norm_act, norm_act=norm_act, act=act, attn=attn, self_attn=self_attn)
    return layer_fn


class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHW',
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'pool_type=' + self.pool_type \
            + ', flatten=' + str(self.flatten) + ')'


def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv, \
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features


def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NCHW',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = nn.Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc


class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            input_fmt: str = 'NCHW',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = nn.Dropout(drop_rate)
        self.fc = fc
        self.flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = nn.Flatten(1) if self.use_conv and pool_type else nn.Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)


class MobileViTXSPyTorch(nn.Module, PyTorchModelHubMixin):
    """
    This class defines a new model variant called MobileViTXSPyTorch.
    It is derived from the timm/ByobNet class but is tailored to utilize the mobilevit-xs configuration by default.
    """

    def __init__(self):
        super().__init__()
        variant = 'mobilevit_xs'
        cfg = model_cfgs[variant]
        num_classes = 1000
        in_chans = 3
        global_pool = 'avg'
        output_stride = 32
        drop_rate = 0.
        drop_path_rate = 0.
        zero_init_last = True

        self.num_classes = num_classes
        self.drop_rate = drop_rate

        layers = get_layer_fns(cfg)
        feat_size = None

        self.feature_info = []
        stem_chs = int(round((cfg.stem_chs or cfg.blocks[0].c) * cfg.width_factor))
        self.stem, stem_feat = create_byob_stem(in_chans, stem_chs, cfg.stem_type, cfg.stem_pool, layers=layers)
        self.feature_info.extend(stem_feat[:-1])
        feat_size = reduce_feat_size(feat_size, stride=stem_feat[-1]['reduction'])

        self.stages, stage_feat = create_byob_stages(
            cfg,
            drop_path_rate,
            output_stride,
            stem_feat[-1],
            layers=layers,
            feat_size=feat_size,
        )
        self.feature_info.extend(stage_feat[:-1])

        prev_chs = stage_feat[-1]['num_chs']
        if cfg.num_features:
            self.num_features = int(round(cfg.width_factor * cfg.num_features))
            self.final_conv = layers.conv_norm_act(prev_chs, self.num_features, 1)
        else:
            self.num_features = prev_chs
            self.final_conv = nn.Identity()
        self.feature_info += [
            dict(num_chs=self.num_features, reduction=stage_feat[-1]['reduction'], module='final_conv')]

        self.head = ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
        )

        # init weights
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

        # We have defined pretrained_cfg to represent the configuration specific to mobilevit-xs pretrained model,
        # including relevant items for dataset and data loader.
        self.pretrained_cfg = {'input_size': (3, 256, 256),
                               'fixed_input_size': False,
                               'interpolation': 'bicubic',
                               'crop_pct': 0.9,
                               'crop_mode': 'center',
                               'mean': (0.0, 0.0, 0.0),
                               'std': (1.0, 1.0, 1.0),
                               'num_classes': 1000,
                               'pool_size': (8, 8)}

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def save_pretrained(self, save_directory, **kwargs):

        # Call the original save_pretrained method
        super().save_pretrained(save_directory, **kwargs)
