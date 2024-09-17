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
The following code was mostly duplicated from https://github.com/huggingface/pytorch-image-models
and changed to generate an equivalent Keras model.
Main changes:
  * Torch layers replaced with Keras layers
  * removed class inheritance from torch.nn.Module
  * changed "forward" class methods with "__call__"
"""

import types
from functools import partial
import tensorflow as tf

from timm.layers import DropPath, make_divisible

__all__ = [
    'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv', 'InvertedResidual', 'CondConvResidual', 'EdgeResidual']


def handle_name(_name):
    return '' if _name is None or _name == '' else _name


def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


def create_act_layer(act_name, **kwargs):
    if isinstance(act_name, str):
        raise NotImplemented
    elif isinstance(act_name, tf.keras.layers.Layer):
        return act_name(**kwargs)
    else:
        return act_name


def get_attn(attn_type):
    if isinstance(attn_type, tf.keras.layers.Layer):
        return attn_type
    module_cls = None
    if attn_type:
        if isinstance(attn_type, str):
            raise NotImplemented
            attn_type = attn_type.lower()
            # Lightweight attention modules (channel and/or coarse spatial).
            # Typically added to existing network architecture blocks in addition to existing convolutions.
            if attn_type == 'se':
                module_cls = SEModule
            elif attn_type == 'ese':
                module_cls = EffectiveSEModule
            elif attn_type == 'eca':
                module_cls = EcaModule
            elif attn_type == 'ecam':
                module_cls = partial(EcaModule, use_mlp=True)
            elif attn_type == 'ceca':
                module_cls = CecaModule
            elif attn_type == 'ge':
                module_cls = GatherExcite
            elif attn_type == 'gc':
                module_cls = GlobalContext
            elif attn_type == 'gca':
                module_cls = partial(GlobalContext, fuse_add=True, fuse_scale=False)
            elif attn_type == 'cbam':
                module_cls = CbamModule
            elif attn_type == 'lcbam':
                module_cls = LightCbamModule

            # Attention / attention-like modules w/ significant params
            # Typically replace some of the existing workhorse convs in a network architecture.
            # All of these accept a stride argument and can spatially downsample the input.
            elif attn_type == 'sk':
                module_cls = SelectiveKernel
            elif attn_type == 'splat':
                module_cls = SplitAttn

            # Self-attention / attention-like modules w/ significant compute and/or params
            # Typically replace some of the existing workhorse convs in a network architecture.
            # All of these accept a stride argument and can spatially downsample the input.
            elif attn_type == 'lambda':
                return LambdaLayer
            elif attn_type == 'bottleneck':
                return BottleneckAttn
            elif attn_type == 'halo':
                return HaloAttn
            elif attn_type == 'nl':
                module_cls = NonLocalAttn
            elif attn_type == 'bat':
                module_cls = BatNonLocalAttn

            # Woops!
            else:
                assert False, "Invalid attn module (%s)" % attn_type
        elif isinstance(attn_type, bool):
            raise NotImplemented
            if attn_type:
                module_cls = SEModule
        else:
            module_cls = attn_type
    return module_cls


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    s = kwargs.pop('stride', None)
    if s is not None:
        kwargs.update({'strides': s})
    d = kwargs.pop('dilation', None)
    if d is not None:
        kwargs.update({'dilation_rate': d})
    assert padding in ['valid', 'same'], 'Not Implemented'
    kwargs.setdefault('use_bias', kwargs.pop('bias', False))
    if kwargs.get('groups', -1) == in_chs:
        kwargs.pop('groups', None)
        return tf.keras.layers.DepthwiseConv2D(kernel_size, padding=padding, **kwargs)
    else:
        return tf.keras.layers.Conv2D(out_chs, kernel_size, padding=padding, **kwargs)


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop('padding', '')
    padding, is_dynamic = padding.lower(), True
    if is_dynamic:
        if pool_type == 'avg':
            raise NotImplemented
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == 'max':
            # return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
            return tf.keras.layers.MaxPooling2D(kernel_size, strides=stride, padding=padding.lower())
        else:
            assert False, f'Unsupported pool type {pool_type}'
    else:
        raise NotImplemented
        if pool_type == 'avg':
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == 'max':
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f'Unsupported pool type {pool_type}'


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """ Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.

    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        raise NotImplemented
        assert 'num_experts' not in kwargs  # MixNet + CondConv combo not supported currently
        if 'groups' in kwargs:
            groups = kwargs.pop('groups')
            if groups == in_channels:
                kwargs['depthwise'] = True
            else:
                assert groups == 1
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        # for DW out_channels must be multiple of in_channels as must have out_channels % groups == 0
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            raise NotImplemented
            m = CondConv2d(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


class SqueezeExcite:
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=tf.keras.layers.ReLU,
            gate_layer=tf.sigmoid, force_act_layer=None, rd_round_fn=None, name=None):
        name = handle_name(name)
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        # self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.conv_reduce = tf.keras.layers.Conv2D(rd_channels, 1, name=name + 'conv_reduce')
        self.act1 = create_act_layer(act_layer, name=name + 'act1')
        # self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.conv_expand = tf.keras.layers.Conv2D(in_chs, 1, name=name + 'conv_expand')
        self.gate = create_act_layer(gate_layer)

    def __call__(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct:
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """
    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, dilation=1, group_size=0, pad_type='',
            skip=False, act_layer=tf.keras.layers.ReLU, norm_layer=tf.keras.layers.BatchNormalization,
            drop_path_rate=0., name=None):
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs

        self.conv = create_conv2d(
            in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            return dict(module='bn1', hook_type='forward', num_chs=self.conv.filters)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv.filters)

    def __call__(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class DepthwiseSeparableConv:
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=tf.keras.layers.ReLU,
            norm_layer=tf.keras.layers.BatchNormalization, se_layer=None, drop_path_rate=0., name=None):
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type,
            groups=groups, name=name + '/conv_dw')
        self.bn1 = norm_act_layer(in_chs, name=name + '/bn1')

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer, name=name + '/se') if se_layer else None

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type, name=name + '/conv_pw')
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act, name=name + '/bn2')
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else None

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.filters)

    def __call__(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + shortcut
        return x


class InvertedResidual:
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=tf.keras.layers.ReLU,
            norm_layer=tf.keras.layers.BatchNormalization, se_layer=None, conv_kwargs=None, drop_path_rate=0.,
            name=None):
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, name=name + '/conv_pw', **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, name=name + '/bn1')

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            groups=groups, padding=pad_type, name=name + '/conv_dw', **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, name=name + '/bn2')

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else None

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type,
                                      name=name + '/conv_pwl', **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False, name=name + '/bn3')
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else None

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.filters)

    def __call__(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + shortcut
        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=tf.keras.layers.ReLU,
            norm_layer=tf.keras.layers.BatchNormalization, se_layer=None, num_experts=0, drop_path_rate=0.,
            name=None):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super(CondConvResidual, self).__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, group_size=group_size,
            pad_type=pad_type, act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_layer=se_layer, norm_layer=norm_layer, conv_kwargs=conv_kwargs,
            drop_path_rate=drop_path_rate)

        # self.routing_fn = nn.Linear(in_chs, self.num_experts)
        self.routing_fn = tf.keras.layers.Dense(self.num_experts)

    def __call__(self, x):
        shortcut = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EdgeResidual:
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
            self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, group_size=0, pad_type='',
            force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, act_layer=tf.keras.layers.ReLU,
            norm_layer=tf.keras.layers.BatchNormalization, se_layer=None, drop_path_rate=0.,
            name=None):
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, before PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.filters)

    def __call__(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class BatchNormAct2d:
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """
    def __init__(
            self,
            num_features,
            epsilon=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            apply_act=True,
            act_layer=tf.keras.layers.ReLU,
            act_kwargs=None,
            inplace=True,
            drop_layer=None,
            device=None,
            dtype=None,
            name=None
    ):
        assert affine, 'Not Implemented'
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)
        if act_kwargs is None:
            act_kwargs = {}
        self.act = act_layer(**act_kwargs) if apply_act else None

    def __call__(self, x):
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


_NORM_ACT_MAP = dict(batchnorm=BatchNormAct2d)
_NORM_ACT_TYPES = {m for n, m in _NORM_ACT_MAP.items()}
_NORM_ACT_REQUIRES_ARG = {BatchNormAct2d}


def get_norm_act_layer(norm_layer, act_layer=None):
    assert isinstance(norm_layer, (type, str,  types.FunctionType, partial))
    # assert act_layer is None or isinstance(act_layer, (type, str, types.FunctionType, partial))
    norm_act_kwargs = {}

    # unbind partial fn, so args can be rebound later
    if isinstance(norm_layer, partial):
        norm_act_kwargs.update(norm_layer.keywords)
        norm_layer = norm_layer.func

    if isinstance(norm_layer, str):
        raise NotImplemented
        layer_name = norm_layer.replace('_', '').lower().split('-')[0]
        norm_act_layer = _NORM_ACT_MAP.get(layer_name, None)
    elif norm_layer in _NORM_ACT_TYPES:
        norm_act_layer = norm_layer
    elif isinstance(norm_layer,  types.FunctionType):
        raise NotImplemented
        # if function type, must be a lambda/fn that creates a norm_act layer
        norm_act_layer = norm_layer
    else:
        type_name = norm_layer.__name__.lower()
        if type_name.startswith('batchnormalization'):
            norm_act_layer = BatchNormAct2d
        elif type_name.startswith('groupnorm'):
            raise NotImplemented
            norm_act_layer = GroupNormAct
        elif type_name.startswith('groupnorm1'):
            raise NotImplemented
            norm_act_layer = functools.partial(GroupNormAct, num_groups=1)
        elif type_name.startswith('layernorm2d'):
            raise NotImplemented
            norm_act_layer = LayerNormAct2d
        elif type_name.startswith('layernorm'):
            raise NotImplemented
            norm_act_layer = LayerNormAct
        else:
            assert False, f"No equivalent norm_act layer for {type_name}"

    if norm_act_layer in _NORM_ACT_REQUIRES_ARG:
        # pass `act_layer` through for backwards compat where `act_layer=None` implies no activation.
        # In the future, may force use of `apply_act` with `act_layer` arg bound to relevant NormAct types
        norm_act_kwargs.setdefault('act_layer', act_layer)
    if norm_act_kwargs:
        norm_act_layer = partial(norm_act_layer, **norm_act_kwargs)  # bind/rebind args
    return norm_act_layer
