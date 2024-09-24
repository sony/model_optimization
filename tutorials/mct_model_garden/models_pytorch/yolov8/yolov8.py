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
Yolov8n Object Detection Model - PyTorch implementation

This code contains a PyTorch implementation of Yolov8n object detection model, following
https://github.com/ultralytics/ultralytics.

Usage:
  model, cfg_dict = yolov8_pytorch("yolov8n.yaml")
  pretrained_weights = torch.load('/path/to/pretrained/yolov8n.pt')['model'].state_dict()
  model.load_state_dict(pretrained_weights, strict=False)
  model.eval()

Main changes:
  Modify layers to make them more suitable for quantization
  torch.fx compatibility
  Detect head (mainly the box decoding part that was optimized for model quantization)
  Inheritance class from HuggingFace
  Implement box decoding into Detect Layer

Notes and Limitations:
- The model has been tested only with the default settings from Ultralytics, specifically using a 640x640 input resolution and 80 object classes.
- Anchors and strides are hardcoded as constants within the model, meaning they are not included in the weights file from Ultralytics.

The code is organized as follows:
- Classes definitions of Yolov8n building blocks: Conv, Bottleneck, C2f, SPPF, Upsample, Concaat, DFL and Detect
- Detection Model definition: ModelPyTorch
- PostProcessWrapper Wrapping the Yolov8n model with PostProcess layer (Specifically, sony_custom_layers/multiclass_nms)
- A getter function for getting a new instance of the model

For more details on the Yolov8n model, refer to the original repository:
https://github.com/ultralytics/ultralytics

"""
import contextlib
import math
import re
from copy import deepcopy
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from huggingface_hub import PyTorchModelHubMixin
import importlib

from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8_postprocess import postprocess_yolov8_keypoints
if importlib.util.find_spec("sony_custom_layers"):
    from sony_custom_layers.pytorch.object_detection.nms import multiclass_nms


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


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""

        y1 = self.cv1(x).chunk(2, 1)
        y = [y1[0], y1[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i, stride in enumerate(strides):
        h, w = int(feats[i]), int(feats[i])
        sx = torch.arange(end=w, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


class Detect(nn.Module):
    def __init__(self, nc: int = 80,
                 ch: List[int] = ()):
        """
        Detection layer for YOLOv8.

        Args:
            nc (int): Number of classes.
            ch (List[int]): List of channel values for detection layers.

        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.Tensor([8, 16, 32])
        self.feat_sizes = torch.Tensor([80, 40, 20])
        self.img_size = 640  # img size
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3),
                          Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                                               nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(self.feat_sizes,
                                                                    self.stride, 0.5))
        anchors = anchors * strides

        self.register_buffer('anchors', anchors)
        self.register_buffer('strides', strides)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.nc), 1)

        y_cls = cls.sigmoid().transpose(1, 2)

        dfl = self.dfl(box)
        dfl = dfl * self.strides

        # box decoding
        lt, rb = dfl.chunk(2, 1)
        y1 = self.anchors.unsqueeze(0)[:, 0, :] - lt[:, 0, :]
        x1 = self.anchors.unsqueeze(0)[:, 1, :] - lt[:, 1, :]
        y2 = self.anchors.unsqueeze(0)[:, 0, :] + rb[:, 0, :]
        x2 = self.anchors.unsqueeze(0)[:, 1, :] + rb[:, 1, :]
        y_bb = torch.stack((x1, y1, x2, y2), 1).transpose(1, 2)
        return y_bb, y_cls

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Detect_wo_bb_dec(nn.Module):
    def __init__(self, nc: int = 80,
                 ch: List[int] = ()):
        """
        Detection layer for YOLOv8. Bounding box decoding was removed.
        Args:
            nc (int): Number of classes.
            ch (List[int]): List of channel values for detection layers.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.Tensor([8, 16, 32])
        self.feat_sizes = torch.Tensor([80, 40, 20])
        self.img_size = 640  # img size
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3),
                          Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                                               nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.nc), 1)

        y_cls = cls.sigmoid()
        y_bb = self.dfl(box)
        return y_bb, y_cls


    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

class Pose(Detect_wo_bb_dec):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect_wo_bb_dec.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        y_bb, y_cls = self.detect(self, x)
        return y_bb, y_cls, kpt

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (
                Conv,
                Bottleneck,
                SPPF,
                C2f,
                nn.ConvTranspose2d,
        ):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if m in [C2f]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in [Segment, Detect, Pose]:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def initialize_weights(model):
    """Initialize model weights to random values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

def model_predict(model: Any,
                  inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch device and data type,
    and detaches and moves the outputs to the CPU.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    device = get_working_device()
    inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)

    # Run Pytorch inference on the batch
    outputs = model(inputs)

    # Detach outputs and move to cpu
    outputs = outputs.cpu().detach()
    return outputs

class PostProcessWrapper(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 score_threshold: float = 0.001,
                 iou_threshold: float = 0.7,
                 max_detections: int = 300):
        """
        Wrapping PyTorch Module with multiclass_nms layer from sony_custom_layers.

        Args:
            model (nn.Module): Model instance.
            score_threshold (float): Score threshold for non-maximum suppression.
            iou_threshold (float): Intersection over union threshold for non-maximum suppression.
            max_detections (float): The number of detections to return.
        """
        super(PostProcessWrapper, self).__init__()
        self.model = model
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def forward(self, images):
        # model inference
        outputs = self.model(images)

        boxes = outputs[0]
        scores = outputs[1]
        nms = multiclass_nms(boxes=boxes, scores=scores, score_threshold=self.score_threshold,
                             iou_threshold=self.iou_threshold, max_detections=self.max_detections)
        return nms

def keypoints_model_predict(model: Any, inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch device and data type,
    and detaches and moves the outputs to the CPU.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    device = get_working_device()
    inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)

    # Run Pytorch inference on the batch
    outputs = model(inputs)

    # Detach outputs and move to cpu
    output_np = [o.detach().cpu().numpy() for o in outputs]

    return postprocess_yolov8_keypoints(output_np)

def seg_model_predict(model: Any,
                  inputs: np.ndarray) -> List:
    """
    Perform inference using the provided PyTorch model on the given inputs.

    This function handles moving the inputs to the appropriate torch data type and format,
    and returns the outputs.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        List: List containing tensors of predictions.
    """
    input_tensor = torch.from_numpy(inputs).unsqueeze(0)  # Add batch dimension
    device = get_working_device()
    input_tensor = input_tensor.to(device)
    # Run the model
    with torch.no_grad():
        outputs = model(input_tensor)
    outputs = [output.cpu() for output in outputs]
    return outputs

def yolov8_pytorch(model_yaml: str) -> (nn.Module, Dict):
    """
    Create PyTorch model of YOLOv8 detection.

    Args:
        model_yaml (str): Name of the YOLOv8 model configuration file (YAML format).

    Returns:
        model: YOLOv8 detection model.
        cfg_dict: YOLOv8 detection model configuration dictionary.
    """
    cfg = model_yaml
    cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
    model = ModelPyTorch(cfg_dict)  # model
    return model, cfg_dict


def yolov8_pytorch_pp(model_yaml: str,
                      score_threshold: float = 0.001,
                      iou_threshold: float = 0.7,
                      max_detections: int = 300) -> (nn.Module, Dict):
    """
    Create PyTorch model of YOLOv8 detection with PostProcess.

    Args:
        model_yaml (str): Name of the YOLOv8 model configuration file (YAML format).
        score_threshold (float): Score threshold for non-maximum suppression.
        iou_threshold (float): Intersection over union threshold for non-maximum suppression.
        max_detections (float): The number of detections to return.

    Returns:
        model: YOLOv8_pp detection model.
        cfg_dict: YOLOv8_pp detection model configuration dictionary.
    """
    cfg = model_yaml
    cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
    model = ModelPyTorch(cfg_dict)  # model
    model_pp = PostProcessWrapper(model=model,
                                  score_threshold=score_threshold,
                                  iou_threshold=iou_threshold,
                                  max_detections=max_detections)
    return model_pp, cfg_dict

class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        y_bb, y_cls = self.detect(self, x)

        return y_bb, y_cls, mc, p


class ModelPyTorch(nn.Module, PyTorchModelHubMixin):
    """
    Unified YOLOv8 model for both detection and segmentation.

    Args:
        cfg (dict): Model configuration in the form of a YAML string or a dictionary.
        ch (int): Number of input channels.
        mode (str): Mode of operation ('detection' or 'segmentation').
    """
    def __init__(self, cfg: dict, ch: int = 3, mode: str = 'detection'):
        super().__init__()
        self.yaml = cfg
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        self.mode = mode
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch)
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Segment) and self.mode == 'segmentation':
            m.inplace = self.inplace
            m.bias_init()
        elif isinstance(m, Detect) and self.mode == 'detection':
            m.inplace = self.inplace
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        initialize_weights(self)

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def make_tensors_contiguous(self):
        for name, param in self.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        for name, buffer in self.named_buffers():
            if not buffer.is_contiguous():
                buffer.data = buffer.data.contiguous()

    def save_pretrained(self, save_directory, **kwargs):
        # Make tensors contiguous
        self.make_tensors_contiguous()
        # Call the original save_pretrained method
        super().save_pretrained(save_directory, **kwargs)
