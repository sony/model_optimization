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

from functools import partial
from typing import Any, Dict, Optional, Union, List
import tensorflow as tf
from timm.models import parse_model_name, split_model_name_tag, is_model, build_model_with_cfg, FeatureInfo
from timm.models._efficientnet_builder import BN_EPS_TF_DEFAULT, decode_arch_def, round_channels
from timm.models._builder import pretrained_cfg_for_features

from tutorials.mct_model_garden.models_keras.efficientdet.effnet_blocks_keras import create_conv2d, SqueezeExcite, get_attn, \
    handle_name, get_norm_act_layer, CondConvResidual, InvertedResidual, DepthwiseSeparableConv, EdgeResidual, \
    ConvBnAct, BatchNormAct2d


__all__ = ["EfficientNetBuilder", "decode_arch_def", "efficientnet_init_weights",
           'resolve_bn_args', 'resolve_act_layer', 'round_channels', 'BN_MOMENTUM_TF_DEFAULT', 'BN_EPS_TF_DEFAULT']


# #######################################################################################
# This file generates the Keras model. It's based on the EfficientNet code in the timm
# repository, and switched the Torch Modules with Keras layers
# #######################################################################################


def _log_info_if(_str, _versbose):
    if _versbose:
        print(_str)


class EfficientNetBuilder:
    """ Build Trunk Blocks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """
    def __init__(self, output_stride=32, pad_type='', round_chs_fn=round_channels, se_from_exp=False,
                 act_layer=None, norm_layer=None, se_layer=None, drop_path_rate=0., feature_location=''):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp  # calculate se channel reduction from expanded (mid) chs
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = get_attn(se_layer)
        try:
            self.se_layer(8, rd_ratio=1.0)  # test if attn layer accepts rd_ratio arg
            self.se_has_ratio = True
        except TypeError:
            self.se_has_ratio = False
        self.drop_path_rate = drop_path_rate
        if feature_location == 'depthwise':
            # old 'depthwise' mode renamed 'expansion' to match TF impl, old expansion mode didn't make sense
            # _logger.warning("feature_location=='depthwise' is deprecated, using 'expansion'")
            feature_location = 'expansion'
        self.feature_location = feature_location
        assert feature_location in ('bottleneck', 'expansion', '')
        self.verbose = False

        # state updated during build, consumed by model
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count, name):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop('block_type')
        ba['name'] = name
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self.round_chs_fn(ba['out_chs'])
        if 'force_in_chs' in ba and ba['force_in_chs']:
            # NOTE this is a hack to work around mismatch in TF EdgeEffNet impl
            ba['force_in_chs'] = self.round_chs_fn(ba['force_in_chs'])
        ba['pad_type'] = self.pad_type
        # block act fn overrides the model default
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'] is not None else self.act_layer
        assert ba['act_layer'] is not None
        ba['norm_layer'] = self.norm_layer
        ba['drop_path_rate'] = drop_path_rate
        if bt != 'cn':
            se_ratio = ba.pop('se_ratio')
            if se_ratio and self.se_layer is not None:
                if not self.se_from_exp:
                    # adjust se_ratio by expansion ratio if calculating se channels from block input
                    se_ratio /= ba.get('exp_ratio', 1.0)
                if self.se_has_ratio:
                    ba['se_layer'] = partial(self.se_layer, rd_ratio=se_ratio)
                else:
                    ba['se_layer'] = self.se_layer

        if bt == 'ir':
            if self.verbose:
                print('  InvertedResidual {}, Args: {}'.format(block_idx, str(ba)))
            block = CondConvResidual(**ba) if ba.get('num_experts', 0) else InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            if self.verbose:
                print('  DepthwiseSeparable {}, Args: {}'.format(block_idx, str(ba)))
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            _log_info_if('  EdgeResidual {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            _log_info_if('  ConvBnAct {}, Args: {}'.format(block_idx, str(ba)), self.verbose)
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt

        self.in_chs = ba['out_chs']  # update in_chs for arg of next block
        return block

    def __call__(self, in_chs, model_block_args, name=None):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            model_block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        name = handle_name(name)
        if self.verbose:
            print('Building model trunk with %d stages...' % len(model_block_args))
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]['stride'] > 1:
            # if the first block starts with a stride, we need to extract first level feat from stem
            feature_info = dict(module='bn1', num_chs=in_chs, stage=0, reduction=current_stride)
            self.features.append(feature_info)

        # outer list of block_args defines the stacks
        for stack_idx, stack_args in enumerate(model_block_args):
            last_stack = stack_idx + 1 == len(model_block_args)
            if self.verbose:
                print('Stack: {}'.format(stack_idx))
            assert isinstance(stack_args, list)

            blocks = []
            # each stack (stage of blocks) contains a list of block arguments
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                if self.verbose:
                    print(' Block: {}'.format(block_idx))

                assert block_args['stride'] in (1, 2)
                if block_idx >= 1:   # only the first block in any stack can have a stride > 1
                    block_args['stride'] = 1

                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = next_stack_idx >= len(model_block_args) or \
                        model_block_args[next_stack_idx][0]['stride'] > 1

                next_dilation = current_dilation
                if block_args['stride'] > 1:
                    next_output_stride = current_stride * block_args['stride']
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args['stride']
                        block_args['stride'] = 1
                        if self.verbose:
                            print('  Converting stride to dilation to maintain output_stride=={}'.format(
                            self.output_stride))
                    else:
                        current_stride = next_output_stride
                block_args['dilation'] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation

                # create the block
                block = self._make_block(block_args, total_block_idx, total_block_count, f'{name}/{stack_idx}/{block_idx}')
                blocks.append(block)

                # stash feature module name and channel info for model feature extraction
                if extract_features:
                    feature_info = dict(
                        stage=stack_idx + 1,
                        reduction=current_stride,
                        **block.feature_info(self.feature_location),
                    )
                    leaf_name = feature_info.get('module', '')
                    if leaf_name:
                        feature_info['module'] = '/'.join([f'blocks.{stack_idx}.{block_idx}', leaf_name])
                    else:
                        assert last_block
                        feature_info['module'] = f'blocks.{stack_idx}'
                    self.features.append(feature_info)

                total_block_idx += 1  # incr global block idx (across all stacks)
            stages.append(blocks)
        return stages


class EfficientNetFeatures:
    """ EfficientNet Feature Extractor

    A work-in-progress feature extraction module for EfficientNet, to use as a backbone for segmentation
    and object detection models.
    """

    def __init__(
            self,
            block_args,
            out_indices=(0, 1, 2, 3, 4),
            feature_location='bottleneck',
            in_chans=3,
            stem_size=32,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
            name=None
    ):
        name = handle_name(name)
        act_layer = act_layer or tf.keras.layers.ReLU
        norm_layer = norm_layer or tf.keras.layers.BatchNormalization
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type, name=name + '/conv_stem')
        self.bn1 = norm_act_layer(stem_size, name=name + '/bn1')

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            feature_location=feature_location,
        )
        self.blocks = builder(stem_size, block_args, name=name + '/blocks')
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {f['stage']: f['index'] for f in self.feature_info.get_dicts()}

        # efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            raise NotImplemented
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
        #     self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def __call__(self, x) -> List[tf.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                for bb in b:
                    # print(i, type(b), type(bb))
                    x = bb(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_effnet(variant, pretrained=False, **kwargs):
    features_mode = ''
    model_cls = None  # EfficientNet
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        if 'feature_cfg' in kwargs:
            features_mode = 'cfg'
        else:
            kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'global_pool')
            model_cls = EfficientNetFeatures
            features_mode = 'cls'
    else:
        raise NotImplemented

    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        features_only=features_mode == 'cfg',
        pretrained_strict=features_mode != 'cls',
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    if features_mode == 'cls':
        model.pretrained_cfg = model.default_cfg = pretrained_cfg_for_features(model.pretrained_cfg)
    return model


def resolve_bn_args(kwargs):
    bn_args = {}
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['epsilon'] = bn_eps
    return bn_args


def resolve_act_layer(kwargs, default='relu'):
    act_name = kwargs.pop('act_layer', default)
    if act_name == 'relu':
        return tf.keras.layers.ReLU
    elif act_name == 'relu6':
        return partial(tf.keras.layers.ReLU, max_value=6.0)
    else:
        raise NotImplemented


def _gen_efficientnet_lite(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    """Creates an EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16'],
        ['ir_r2_k3_s2_e6_c24'],
        ['ir_r2_k5_s2_e6_c40'],
        ['ir_r3_k3_s2_e6_c80'],
        ['ir_r3_k5_s1_e6_c112'],
        ['ir_r4_k5_s2_e6_c192'],
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=resolve_act_layer(kwargs, 'relu6'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(tf.keras.layers.BatchNormalization, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet(variant, pretrained, **model_kwargs)
    return model


def tf_efficientnet_lite0(pretrained=False, **kwargs):
    """ EfficientNet-Lite0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    kwargs['name'] = 'backbone'
    model = _gen_efficientnet_lite(
        'tf_efficientnet_lite0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model


model_entrypoints = {'tf_efficientnet_lite0': tf_efficientnet_lite0}


def create_model(
        model_name: str,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], Any]] = None,
        pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '',
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        **kwargs,
):
    """Create a model.

    Lookup model's entrypoint function and pass relevant args to create a new model.

    <Tip>
        **kwargs will be passed through entrypoint fn to ``timm.models.build_model_with_cfg()``
        and then the model class __init__(). kwargs values set to None are pruned before passing.
    </Tip>

    Args:
        model_name: Name of model to instantiate.
        pretrained: If set to `True`, load pretrained ImageNet-1k weights.
        pretrained_cfg: Pass in an external pretrained_cfg for model.
        pretrained_cfg_overlay: Replace key-values in base pretrained_cfg with these.
        checkpoint_path: Path of checkpoint to load _after_ the model is initialized.
        scriptable: Set layer config so that model is jit scriptable (not working for all models yet).
        exportable: Set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet).
        no_jit: Set layer config so that model doesn't utilize jit scripted layers (so far activations only).

    Keyword Args:
        drop_rate (float): Classifier dropout rate for training.
        drop_path_rate (float): Stochastic depth drop rate for training.
        global_pool (str): Classifier global pooling type.

    Example:

    ```py
    >>> from timm import create_model

    >>> # Create a MobileNetV3-Large model with no pretrained weights.
    >>> model = create_model('mobilenetv3_large_100')

    >>> # Create a MobileNetV3-Large model with pretrained weights.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True)
    >>> model.num_classes
    1000

    >>> # Create a MobileNetV3-Large model with pretrained weights and a new head with 10 classes.
    >>> model = create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
    >>> model.num_classes
    10
    ```
    """
    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    model_source, model_name = parse_model_name(model_name)
    if model_source == 'hf-hub':
        assert not pretrained_cfg, 'pretrained_cfg should not be set when sourcing model from Hugging Face Hub.'
        # For model names specified in the form `hf-hub:path/architecture_name@revision`,
        # load model weights + pretrained_cfg from Hugging Face hub.
        raise NotImplemented
        # pretrained_cfg, model_name = load_model_config_from_hf(model_name)
    else:
        model_name, pretrained_tag = split_model_name_tag(model_name)
        if pretrained_tag and not pretrained_cfg:
            # a valid pretrained_cfg argument takes priority over tag in model name
            pretrained_cfg = pretrained_tag

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoints[model_name]
    # with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
    model = create_fn(
        pretrained=pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
        **kwargs,
    )

    return model
