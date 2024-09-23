# ------------------------------------------------------------------------------
# This file contains code from the Ultralytics repository (YOLOv8)
# Copyright (C) 2024  Ultralytics
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------

"""
Yolov8n Object Detection Model - Keras implementation

This code contains a TensorFlow/Keras implementation of Yolov8n object detection model, following
https://github.com/ultralytics/ultralytics. This implementation includes a slightly modified version of yolov8
detection-head (mainly the box decoding part) that was optimized for model quantization.

The code is organized as follows:
- Classes definitions of Yolov8n building blocks: Conv, Bottleneck, C2f, SPPF, Upsample, Concaat, DFL and Detect
- Detection Model definition: DetectionModelKeras
- A getter function for getting a new instance of the model

For more details on the Yolov8n model, refer to the original repository:
https://github.com/ultralytics/ultralytics

"""
import sys
from pathlib import Path
import re
import yaml
from copy import deepcopy
import contextlib
import math
import numpy as np
import tensorflow as tf
from keras import layers, initializers
from keras.layers import BatchNormalization, Concatenate, UpSampling2D, Input
from keras.models import Model
from typing import Dict, Optional, List, Tuple, Union
import cv2

def yaml_load(file: str = 'data.yaml', append_filename: bool = False) -> Dict[str, any]:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

def make_divisible(x: int, divisor: int) -> int:
    """
    Returns the nearest integer to 'x' that is divisible by 'divisor'.

    Args:
        x (int): The input integer.
        divisor (int): The divisor for which 'x' should be divisible.

    Returns:
        int: The nearest integer to 'x' that is divisible by 'divisor'.
    """
    return math.ceil(x / divisor) * divisor

class Conv:
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, name: str = '', g: int = 1, d: int = 1):
        """
        Standard convolution layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size. Default is 1.
            s (int, optional): Stride. Default is 1.
            name (str, optional): Name of the layer. Default is an empty string.
            g (int, optional): Groups. Default is 1.
            d (int, optional): Dilation rate. Default is 1.

        """
        kernel_size = k[0] if isinstance(k,tuple) else k
        if kernel_size > 1:
            pad = ((1,0), (1,0)) if s > 1 else (1,1)
        else:
            pad = (0,0)
        self.padding2d = layers.ZeroPadding2D(padding=pad)
        self.conv = layers.Conv2D(c2, k, s, 'valid', groups=g, dilation_rate=d, use_bias=False, name=name+'.conv')
        self.bn = layers.BatchNormalization(momentum=0.97, epsilon=1e-3, name=name+'.bn')
        self.act = tf.nn.silu  # default activation
        self.c1 = c1 # Unused in Keras implementation

    def __call__(self, x):
        return self.act(self.bn(self.conv(self.padding2d(x))))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck:
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: Tuple[int, int] = (3, 3), e: float = 0.5,
                 name: str = ''):
        """
        Standard bottleneck layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            shortcut (bool, optional): Use shortcut connection. Default is True.
            g (int, optional): Groups. Default is 1.
            k (Tuple[int, int], optional): Kernel sizes. Default is (3, 3).
            e (float, optional): Hidden channels ratio. Default is 0.5.
            name (str, optional): Name of the layer. Default is an empty string.

        """
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1, name=f'{name}.cv1')
        self.cv2 = Conv(c_, c2, k[1], 1, g=g, name=f'{name}.cv2')
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f:
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, name: str = '', g: int = 1, e: float = 0.5):
        """
        CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int, optional): Number of Bottleneck blocks. Default is 1.
            shortcut (bool, optional): Use shortcut connection. Default is False.
            name (str, optional): Name of the layer. Default is an empty string.
            g (int, optional): Groups. Default is 1.
            e (float, optional): Hidden channels ratio. Default is 0.5.

        """
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, name=f'{name}.cv1')
        self.cv2 = Conv((2 + n) * self.c, c2, 1, name=f'{name}.cv2')  # optional act=FReLU(c2)
        self.m = [Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, name=f'{name}.m.{i}') for i in range(n)]

    def __call__(self, x):
        y1 = self.cv1(x)
        y = tf.split(y1, 2, -1)
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(Concatenate(axis=-1)(y))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF:
    def __init__(self, c1: int, c2: int, k: int = 5, name: str = ''):
        """
        Spatial Pyramid Pooling - Fast (SPPF) layer.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int, optional): Kernel size. Default is 5.
            name (str, optional): Name of the layer. Default is an empty string.
        """
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, name=f'{name}.cv1')
        self.cv2 = Conv(c_ * 4, c2, 1, 1, name=f'{name}.cv2')
        if k // 2 == 0:
            padding = 'valid'
        else:
            padding = 'same'
        self.m = layers.MaxPooling2D(pool_size=k, strides=1, padding=padding)

    def __call__(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = Concatenate()([x, y1, y2, y3])
        return self.cv2(y)
        

class Upsample:
    def __init__(self, size: Tuple[int, int] = None, scale_factor: Tuple[float, float] = None, mode: str = 'nearest'):
        """
        Upsample layer using `UpSampling2D` for resizing the input.

        Args:
            size (Tuple[int, int], optional): The target size (height, width) for upsampling. Default is None.
            scale_factor (Tuple[float, float], optional): The scaling factor (height_scale, width_scale) for upsampling. Default is None.
            mode (str, optional): The interpolation mode. Default is 'nearest'.
        """
        self.m = UpSampling2D(size=scale_factor, data_format=None, interpolation=mode)

    def __call__(self, x):
        return self.m(x)

class Concat:
    def __init__(self, dimension: int = 1):
        """
        Concatenate a list of tensors along the specified dimension.

        Args:
            dimension (int, optional): The dimension along which tensors will be concatenated. Default is 1.
        """
        self.d = -1 if dimension==1 else dimension

    def __call__(self, x):
        return Concatenate(self.d)(x)

class DFL:
    def __init__(self, c1: int = 8, name: str = ''):
        """
        Distributed focal loss calculation.

        Args:
            c1 (int, optional): The number of classes. Default is 8.
            name (str, optional): Name prefix for layers. Default is an empty string.
        """
        self.c1 = c1
        w = np.expand_dims(np.expand_dims(np.expand_dims(np.arange(c1), 0), 0), -1)
        self.conv = layers.Conv2D(1, 1, use_bias=False, weights=[w], name=f'{name}.dfl.conv')

    def __call__(self, x):
        x_shape = x.shape
        x = tf.reshape(x, (-1, x_shape[1], 4, self.c1))
        x = layers.Softmax(-1)(x)
        return tf.squeeze(self.conv(x), -1)

def make_anchors(feats: List[int], strides: List[int], grid_cell_offset: float = 0.5):
    """
    Generate anchors from features.

    Args:
        feats (List[int]): List of feature sizes for generating anchors.
        strides (List[int]): List of stride values corresponding to each feature size.
        grid_cell_offset (float, optional): Grid cell offset. Default is 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the anchor points and stride tensors.
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = np.arange(stop=w) + grid_cell_offset  # shift x
        sy = np.arange(stop=h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points, dtype='float32'), np.concatenate(stride_tensor, dtype='float32')

def dist2bbox(points: tf.Tensor, distance: tf.Tensor) -> tf.Tensor:
    """
    Decode distance prediction to bounding box.

    Args:
        points (tf.Tensor): Shape (n, 2), [x, y].
        distance (tf.Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).

    Returns:
        tf.Tensor: Decoded bboxes.
    """
    d0, d1, d2, d3 = tf.unstack(distance, 4, -1)
    a0, a1 = tf.unstack(points, 2, -1)
    x1 = layers.ReLU()(tf.math.subtract(a0, d0))  # Adding a relu in order to force unsigned output (which is expected in this case)
    y1 = layers.ReLU()(tf.math.subtract(a1, d1))
    x2 = layers.ReLU()(tf.math.add(a0, d2))
    y2 = layers.ReLU()(tf.math.add(a1, d3))
    return tf.stack([y1, x1, y2, x2], -1)

class Detect:
    def __init__(self, nc: int = 80, ch: List[int] = (), name: str = ''):
        """
        Detection layer for YOLOv8.

        Args:
            nc (int): Number of classes.
            ch (List[int]): List of channel values for detection layers.
            name (str): Name for the detection layer.

        """
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.feat_sizes = [80, 40, 20]
        self.stride_sizes = [8, 16, 32]
        img_size = 640
        nd0, nd1, nd2 = np.cumsum([sz ** 2 for sz in self.feat_sizes]) # split per stride/resolution level

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels

        # Bias initialization for the detection head kernels
        a, b = [], []
        for s in self.stride_sizes:
            a.append(initializers.Constant(1.0))  # box
            b.append(initializers.Constant(math.log(5 / self.nc / (img_size / s) ** 2)))  # cls (.01 objects, 80 classes, 640 img)

        # Detection head
        self.cv2 = [[Conv(x, c2, 3, name=f'{name}.cv2.{i}.0'), Conv(c2, c2, 3, name=f'{name}.cv2.{i}.1'), layers.Conv2D(4 * self.reg_max, 1, bias_initializer=a[i], name=f'{name}.cv2.{i}.2')] for i, x in enumerate(ch)]
        self.cv3 = [[Conv(x, c3, 3, name=f'{name}.cv3.{i}.0'), Conv(c3, c3, 3, name=f'{name}.cv3.{i}.1'), layers.Conv2D(self.nc, 1, bias_initializer=b[i], name=f'{name}.cv3.{i}.2')] for i, x in enumerate(ch)]

        # Distributed Focal Loss
        self.dfl = DFL(self.reg_max, name=name)

        # Yolov8 anchors preparation. The strides are used to scale the different resolution levels
        self.anchors, self.strides = (x.transpose(0,1) for x in make_anchors(self.feat_sizes, self.stride_sizes, 0.5))

        #  Anchors normalization - optimizations for better quantization
        self.strides = self.strides / img_size
        self.split_strides = [self.strides[0], self.strides[nd0], self.strides[nd1]]
        self.anchors = self.anchors * self.strides

    def __call__(self, x):
        # Detection head convolutions. Output per stride level
        feat = self.feat_sizes
        xbox, xcls = [0,0,0], [0,0,0]
        for i in range(self.nl):
            x0 = self.cv2[i][0](x[i])
            x0 = self.cv2[i][1](x0)
            x0 = self.cv2[i][2](x0)
            x1 = self.cv3[i][0](x[i])
            x1 = self.cv3[i][1](x1)
            x1 = self.cv3[i][2](x1)
            xbox[i], xcls[i] = x0, x1

        # Classes - concatenation of the stride levels and sigmoid operator
        cls = Concatenate(axis=1)([tf.reshape(xi, (-1, feat[i] ** 2, self.nc)) for i, xi in enumerate(xcls)])
        y_cls = tf.math.sigmoid(cls)

        # Boxes - DFL operator, stride scaling and lastly concatenation (for better quantization, we want a concatenation of inputs with the same scale)
        box = [tf.reshape(xi, (-1, feat[i] ** 2, self.reg_max * 4)) for i, xi in enumerate(xbox)]
        dist = Concatenate(axis=1)([tf.math.multiply(self.dfl(b), self.split_strides[i]) for i,b in enumerate(box)])
        anchors = tf.expand_dims(self.anchors, 0)
        y_bb = dist2bbox(anchors, dist)
        y_bb = tf.expand_dims(y_bb, 2)

        return [y_bb, y_cls]

def parse_model(d: dict, ch: List[int], verbose: bool = True) -> Tuple[List, List[int]]:
    """
    Parse a YOLO model.yaml dictionary and construct the model architecture.

    Args:
        d (dict): YOLO model.yaml dictionary containing model configuration.
        ch (List[int]): List of initial channel sizes.
        verbose (bool, optional): Verbose mode for printing model details. Default is True.

    Returns:
        list: A list of model layers.
        list: A list of save indices for layers.
    """
    nc, gd, gw, act = d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = 'Upsample' if m == 'nn.Upsample' else m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, C2f, SPPF]:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C2f]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is BatchNormalization:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        if n > 1:
            raise Exception("Not implemented")

        if m is C2f and len(args) == 3:
            args.append(False)
        if m in [Conv, C2f, SPPF, Detect]:
            args.append(f'model.{i}')
        m_ = m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        print(f'{i:>3}{str(f):>20}{n_:>3}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return layers, sorted(save)

class DetectionModelKeras:
    def __init__(self, cfg: dict, ch: int = 3, verbose: bool = True):
        """
        YOLOv8 detection model.

        Args:
            cfg (dict): Model configuration in the form of a YAML string or a dictionary.
            ch (int): Number of input channels.
            verbose (bool, optional): Verbose mode for printing model details. Default is True.
        """
        # Define model
        self.yaml = cfg
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist

    def __call__(self, x):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

def yolov8_keras(model_yaml: str, img_size: int) -> Model:
    """
    Create Keras model of YOLOv8 detection.

    Args:
        model (str): Name of the YOLOv8 model configuration file (YAML format).
        img_size (int): Size of the input image (assuming square dimensions).

    Returns:
        Model: YOLOv8 detection model.
    """
    cfg = model_yaml
    cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
    model_func = DetectionModelKeras(cfg_dict, verbose=True)  # model functionality
    inputs = Input(shape=(img_size, img_size, 3))
    return Model(inputs, model_func(inputs))
