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
The following code was mostly duplicated from https://github.com/rwightman/efficientdet-pytorch
and changed to generate an equivalent Keras model.
Main changes:
  * Torch layers replaced with Keras layers
  * removed class inheritance from torch.nn.Module
  * changed "forward" class methods with "__call__"
"""

import logging
from collections import OrderedDict
from functools import partial
from typing import List, Optional, Union, Tuple

import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


from effdet.anchors import Anchors, get_feat_sizes
from effdet.config import get_fpn_config, set_config_readonly
from effdet.efficientdet import get_feature_info
from tutorials.mct_model_garden.models_keras.efficientdet.effnet_keras import create_model, handle_name
from tutorials.mct_model_garden.models_keras.efficientdet.effnet_blocks_keras import create_conv2d, create_pool2d
from tutorials.mct_model_garden.models_keras.utils.torch2keras_weights_translation import load_state_dict

from sony_custom_layers.keras.object_detection.ssd_post_process import SSDPostProcess
from sony_custom_layers.keras.object_detection import ScoreConverter

_DEBUG = False
_USE_SCALE = False
_ACT_LAYER = tf.nn.swish

# #######################################################################################
# This file generates the Keras model. It's based on the EfficientDet repository in
# https://github.com/rwightman/efficientdet-pytorch, and switched the Torch Modules
# with Keras layers
# #######################################################################################

def get_act_layer(act_type):
    if act_type == 'relu6':
        return partial(tf.keras.layers.ReLU, max_value=6.0)
    else:
        raise NotImplemented


class ConvBnAct2d:
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            norm_layer=tf.keras.layers.BatchNormalization,
            act_layer=_ACT_LAYER,
            name=None
    ):
        name = handle_name(name)
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            name=name + '/conv'
        )
        self.bn = None if norm_layer is None else norm_layer(name=name + '/bn')
        self.act = None if act_layer is None else act_layer()

    def __call__(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d:
    """ Separable Conv
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            channel_multiplier=1.0,
            pw_kernel_size=1,
            norm_layer=tf.keras.layers.BatchNormalization,
            act_layer=_ACT_LAYER,
            name=None
    ):
        name = handle_name(name)
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
            name=name + '/conv_dw'
        )
        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
            name=name + '/conv_pw'
        )
        self.bn = None if norm_layer is None else norm_layer(name=name + '/bn')
        self.act = None if act_layer is None else act_layer()

    def __call__(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Interpolate2d:
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(
            self,
            size: Optional[Union[int, Tuple[int, int]]] = None,
            scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
            mode: str = 'nearest',
            align_corners: bool = False,
    ) -> None:
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

        # tested in keras
        assert self.align_corners in [None, False]
        assert self.scale_factor is None
        if self.mode == 'nearest':
            self.mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        else:
            raise NotImplemented

    def __call__(self, input: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(input, self.size, method=self.mode)


class ResampleFeatureMap:

    def __init__(
            self,
            in_channels,
            out_channels,
            input_size,
            output_size,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=tf.keras.layers.BatchNormalization,
            apply_bn=False,
            redundant_bias=False,
            name=None
    ):
        name = handle_name(name)
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        self.layers = []
        if in_channels != out_channels:
            self.layers.append(ConvBnAct2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias,
                act_layer=None,
                name=f'{name}/conv'#/{len(self.layers)}'
            ))

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ('max', 'avg'):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(downsample, kernel_size=kernel_size, stride=stride, padding=pad_type,
                                          name=name + '/downsample')
            else:
                if _USE_SCALE:  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample, name=name)
            self.layers.append(down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))
                else:
                    self.layers.append(Interpolate2d(size=output_size, mode=upsample))  # 'upsample'

    def __call__(self, x: tf.Tensor) -> List[tf.Tensor]:
        for module in self.layers:
            x = module(x)
        return x


class FpnCombine:
    def __init__(
            self,
            feature_info,
            fpn_channels,
            inputs_offsets,
            output_size,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=tf.keras.layers.BatchNormalization,
            apply_resample_bn=False,
            redundant_bias=False,
            weight_method='attn',
            name=None
    ):
        name = handle_name(name)
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = []  # nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample.append(ResampleFeatureMap(
                feature_info[offset]['num_chs'],
                fpn_channels,
                input_size=feature_info[offset]['size'],
                output_size=output_size,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
                name = name + f'/resample/{offset}'
            ))

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def __call__(self, x: List[tf.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'sum':
            out = tf.keras.layers.Add()(nodes[:2])
            for i in range(2, len(nodes)):
                out = tf.keras.layers.Add()([out, nodes[i]])
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        return out


class Fnode:
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine, after_combine):
        self.combine = combine
        self.after_combine = after_combine

    def __call__(self, x: List[tf.Tensor]) -> tf.Tensor:
        x = self.combine(x)
        for fn in self.after_combine:
            x = fn(x)
        return x


class BiFpnLayer:
    def __init__(
            self,
            feature_info,
            feat_sizes,
            fpn_config,
            fpn_channels,
            num_levels=5,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=tf.keras.layers.BatchNormalization,
            act_layer=_ACT_LAYER,
            apply_resample_bn=False,
            pre_act=True,
            separable_conv=True,
            redundant_bias=False,
            name=None
    ):
        name = handle_name(name)
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc['feat_level']]) for fc in fpn_config.nodes]

        self.fnode = []  # nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info,
                fpn_channels,
                tuple(fnode_cfg['inputs_offsets']),
                output_size=feat_sizes[fnode_cfg['feat_level']],
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg['weight_method'],
                name=f'{name}/fnode/{i}/combine'
            )

            after_combine = []  # nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if pre_act:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.append(act_layer())  # 'act'
            after_combine.append(
                SeparableConv2d(name=f'{name}/fnode/{i}/after_combine/conv', **conv_kwargs) if separable_conv
                else ConvBnAct2d(name=f'{name}/fnode/{i}/after_combine/conv', **conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))

        self.feature_info = fpn_feature_info[-num_levels::]

    def __call__(self, x: List[tf.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class BiFpn:

    def __init__(self, config, feature_info, name):
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or tf.keras.layers.BatchNormalization
        norm_kwargs = {**config.norm_kwargs}
        norm_kwargs['epsilon'] = norm_kwargs.pop('eps', 0.001)
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = []  # nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample.append(ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                    name=name + f'/resample/{level}'
                ))
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size

        self.cell = []  # SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
                name=name + f'/cell/{rep}'
            )
            self.cell.append(fpn_layer)
            feature_info = fpn_layer.feature_info

    def __call__(self, x: List[tf.Tensor]):
        for resample in self.resample:
            x.append(resample(x[-1]))
        for _cell in self.cell:
            x = _cell(x)
        return x


class HeadNet:

    def __init__(self, config, num_outputs, name):
        self.num_levels = config.num_levels
        self.bn_level_first = getattr(config, 'head_bn_level_first', False)
        norm_layer = config.norm_layer or tf.keras.layers.BatchNormalization
        if config.norm_kwargs:
            norm_kwargs = {**config.norm_kwargs}
            if 'eps' in norm_kwargs:
                eps = norm_kwargs.pop('eps')
                norm_kwargs['epsilon'] = eps
            norm_layer = partial(norm_layer, **norm_kwargs)
        act_type = config.head_act_type if getattr(config, 'head_act_type', None) else config.act_type
        act_layer = get_act_layer(act_type) or _ACT_LAYER

        # Build convolution repeats
        conv_fn = SeparableConv2d if config.separable_conv else ConvBnAct2d
        conv_kwargs = dict(
            in_channels=config.fpn_channels,
            out_channels=config.fpn_channels,
            kernel_size=3,
            padding=config.pad_type,
            bias=config.redundant_bias,
            act_layer=None,
            norm_layer=None,
        )
        self.conv_rep = [conv_fn(name=f'{name}/conv_rep/{_}', **conv_kwargs) for _ in range(config.box_class_repeats)]

        # Build batchnorm repeats. There is a unique batchnorm per feature level for each repeat.
        # This can be organized with repeats first or feature levels first in module lists, the original models
        # and weights were setup with repeats first, levels first is required for efficient torchscript usage.
        self.bn_rep = []  # nn.ModuleList()
        if self.bn_level_first:
            for _ in range(self.num_levels):
                self.bn_rep.append([
                    norm_layer(config.fpn_channels, name=f'{name}/bn_rep/{_}/', ) for _ in range(config.box_class_repeats)])
        else:
            for _ in range(config.box_class_repeats):
                self.bn_rep.append([norm_layer(name=f'{name}/bn_rep/{_}/{_level}/bn') for _level in range(self.num_levels)])

        self.act = act_layer

        # Prediction (output) layer. Has bias with special init reqs, see init fn.
        num_anchors = len(config.aspect_ratios) * config.num_scales
        predict_kwargs = dict(
            in_channels=config.fpn_channels,
            out_channels=num_outputs * num_anchors,
            kernel_size=3,
            padding=config.pad_type,
            bias=True,
            norm_layer=None,
            act_layer=None,
            name=f'{name}/predict'
        )
        self.predict = conv_fn(**predict_kwargs)

    def toggle_bn_level_first(self):
        """ Toggle the batchnorm layers between feature level first vs repeat first access pattern
        Limitations in torchscript require feature levels to be iterated over first.

        This function can be used to allow loading weights in the original order, and then toggle before
        jit scripting the model.
        """
        new_bn_rep = []  # nn.ModuleList()
        for i in range(len(self.bn_rep[0])):
            bn_first = []  # nn.ModuleList()
            for r in self.bn_rep.children():
                m = r[i]
                # NOTE original rep first model def has extra Sequential container with 'bn', this was
                # flattened in the level first definition.
                bn_first.append(m[0] if isinstance(m, nn.Sequential) else nn.Sequential(OrderedDict([('bn', m)])))
            new_bn_rep.append(bn_first)
        self.bn_level_first = not self.bn_level_first
        self.bn_rep = new_bn_rep

    def _forward(self, x: List[tf.Tensor]) -> List[tf.Tensor]:
        outputs = []
        for level in range(self.num_levels):
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, self.bn_rep):
                x_level = conv(x_level)
                x_level = bn[level](x_level)  # this is not allowed in torchscript
                x_level = self.act()(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def _forward_level_first(self, x: List[tf.Tensor]) -> List[tf.Tensor]:
        outputs = []
        for level, bn_rep in enumerate(self.bn_rep):  # iterating over first bn dim first makes TS happy
            x_level = x[level]
            for conv, bn in zip(self.conv_rep, bn_rep):
                x_level = conv(x_level)
                x_level = bn(x_level)
                x_level = self.act()(x_level)
            outputs.append(self.predict(x_level))
        return outputs

    def __call__(self, x: List[tf.Tensor]) -> List[tf.Tensor]:
        if self.bn_level_first:
            return self._forward_level_first(x)
        else:
            return self._forward(x)


class EfficientDetKeras:

    def __init__(self, config, pretrained_backbone=True, alternate_init=False):
        self.config = config
        set_config_readonly(self.config)
        self.backbone = create_model(
            config.backbone_name,
            features_only=True,
            out_indices=self.config.backbone_indices or (2, 3, 4),
            pretrained=pretrained_backbone,
            **config.backbone_args,
        )
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(self.config, feature_info, 'fpn')
        self.class_net = HeadNet(self.config, num_outputs=self.config.num_classes, name='class_net')
        self.box_net = HeadNet(self.config, num_outputs=4, name='box_net')

    def toggle_head_bn_level_first(self):
        """ Toggle the head batchnorm layers between being access with feature_level first vs repeat
        """
        self.class_net.toggle_bn_level_first()
        self.box_net.toggle_bn_level_first()

    def get_model(self, input_shape, load_state_dict_to_model=True):
        _input = tf.keras.layers.Input(shape=input_shape)
        x = self.backbone(_input)
        x = self.fpn(x)
        x_class = self.class_net(x)
        x_box = self.box_net(x)

        x_class = [tf.keras.layers.Reshape((-1, self.config.num_classes))(_x) for _x in x_class]
        x_class = tf.keras.layers.Concatenate(axis=1)(x_class)
        x_box = [tf.keras.layers.Reshape((-1, 4))(_x) for _x in x_box]
        x_box = tf.keras.layers.Concatenate(axis=1)(x_box)

        anchors = tf.constant(Anchors.from_config(self.config).boxes.detach().cpu().numpy())

        ssd_pp = SSDPostProcess(anchors, [1, 1, 1, 1], [*self.config.image_size],
                                ScoreConverter.SIGMOID, score_threshold=0.001, iou_threshold=0.5,
                                max_detections=self.config.max_det_per_image)
        outputs = ssd_pp((x_box, x_class))

        model = tf.keras.Model(inputs=_input, outputs=outputs)
        if load_state_dict_to_model:
            load_state_dict(model, self.config.url)
        return model
