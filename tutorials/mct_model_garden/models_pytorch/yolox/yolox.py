# ------------------------------------------------------------------------------
# This file contains code from the https://github.com/Megvii-BaseDetection/YOLOX repository.
# Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

# The following code was mostly duplicated from https://github.com/Megvii-BaseDetection/YOLOX
# and changed to generate an equivalent PyTorch model suitable for quantization.
# Main changes:
#   * Modify layers to make them more suitable for quantization.
#   * Integrate box decoding and NMS into the model
# ==============================================================================
"""

This code contains a PyTorch implementation of Yolox object detection model.
This implementation includes a slightly modified version of Yolox
detection-head (mainly the box decoding part) which is optimized for model quantization.
For more details on Yolox, refer to the original repository:
https://github.com/Megvii-BaseDetection/YOLOX

"""
from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np
from model_compression_toolkit.core.pytorch.pytorch_device_config import get_working_device
from .darknet import CSPDarknet, CSPLayer, BaseConv, DWConv
from sony_custom_layers.pytorch import multiclass_nms, FasterRCNNBoxDecode


class YOLOPAFPN(nn.Module):
    """
    Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, inputs):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(inputs)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.device = get_working_device()
        self.num_classes = num_classes
        self.strides = strides
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def forward(self, xin):
        outputs = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        # [batch, n_anchors_all, 85]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
        return outputs


class YOLOX(nn.Module):
    """
    YOLOX model for object detection.

    Args:
        cfg (dict): Model configuration in the form of a dictionary.
    """

    def __init__(self, cfg):
        super(YOLOX, self).__init__()
        self.device = get_working_device()
        self.cfg = cfg
        self.depth = cfg.get("depth")
        self.width = cfg.get("width")
        self.img_size = cfg.get("img_size")
        self.num_classes = cfg.get("num_classes")
        self.act = cfg.get("act")
        self.depthwise = cfg.get("depthwise")
        self.backbone = YOLOPAFPN(
            self.depth, self.width,
            act=self.act, depthwise=self.depthwise,
        )
        self.head = YOLOXHead(
            self.num_classes, self.width,
            act=self.act, depthwise=self.depthwise,
        )
        self.init_weights()

    def load_weights(self, path):
        """
        Load weights to model.
        Args:
            path (str): weight's file path
        """
        sd = torch.load(path, map_location=self.device, weights_only=True)['model']
        self.load_state_dict(sd)

    def init_weights(self):
        """
        Init batchnorm eps and momentum
        """
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
        self.eval().to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor]:
        """
        Inference
        Args:
            x (tensor): input tensor

        Returns:
            tuple containing tensors of boxes and scores
        """
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)
        boxes = outputs[...,:4]
        # Convert from (xc,yc,w,h) to (yc,xc,h,w)
        xc, yc, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        boxes = torch.stack([yc, xc, h, w], dim=-1)
        scores = outputs[..., 5:] * outputs[..., 4:5] # classes * scores
        return boxes, scores


class YOLOXPostProcess(nn.Module):
    """
    Wrapping YoloX with post process functionality: box decoding and multiclass_nms layer from sony_custom_layers.

    Args:
        model (nn.Module): Model instance.
        img_size: (tuple): Image size input of the model.
        score_threshold (float): Score threshold for non-maximum suppression.
        iou_threshold (float): Intersection over union threshold for non-maximum suppression.
        max_detections (int): The number of detections to return.
    """
    def __init__(self,
                 model: nn.Module,
                 img_size: tuple = (416,416),
                 score_threshold: float = 0.001,
                 iou_threshold: float = 0.65,
                 max_detections: int = 200):
        super(YOLOXPostProcess, self).__init__()
        self.device = get_working_device()
        self.model = model
        self.box_decoder = FasterRCNNBoxDecode(anchors=self.create_anchors(img_size),
                                               scale_factors=[1,1,1,1],
                                               clip_window=[0,0,*img_size])
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def create_anchors(self, img_size: Tuple, strides: List = [8, 16, 32]) -> torch.tensor:
        """
        Create anchors for box decoding operation.
        Args:
            img_size: (tuple): Image size input of the model.
            strides (list): strides to bed used in anchors.

        Returns:
            outputs: tesnor of anchors.
        """
        device = get_working_device()
        fmap_grids = []
        fmap_strides = []
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            fmap_grids.append(grid)
            shape = grid.shape[:2]
            fmap_strides.append(torch.full((*shape, 1), stride))

        s = torch.cat(fmap_strides, dim=1).to(device)
        offsets = s * torch.cat(fmap_grids, dim=1).to(device)
        xc, yc = offsets[..., 0:1], offsets[..., 1:2]
        anchors = torch.concat([(2 * yc - s) / 2, (2 * xc - s) / 2, (2 * yc + s) / 2, (2 * xc + s) / 2], dim=-1)
        anchors = anchors.squeeze(0)
        return anchors

    def forward(self, images: torch.tensor) -> Tuple:
        """
        Perform inference on the given images.
        Args:
            images (np.ndarray): Input data to perform inference on.

        Returns:
            predictions consit of boxes, scores, labels
        """
        # Inference
        boxes, scores = self.model(images)
        # Box decoder
        boxes = self.box_decoder(boxes)
        # NMS
        nms_out = multiclass_nms(boxes=boxes,
                                 scores=scores,
                                 score_threshold=self.score_threshold,
                                 iou_threshold=self.iou_threshold,
                                 max_detections=self.max_detections)
        return nms_out.boxes, nms_out.scores, nms_out.labels


def model_predict(model: nn.Module,
                  inputs: np.ndarray) -> Tuple[torch.tensor]:
    """
    Perform inference using the provided model on the given inputs.

    This function handles moving the inputs to the appropriate torch device and data type,
     detaches and moves the outputs to the CPU.

    Args:
        model (Any): The PyTorch model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        outputs: tuple containing tensors of predictions.
    """
    device = get_working_device()
    inputs = torch.from_numpy(inputs).to(device=device, dtype=torch.float)

    # Run Pytorch inference on the batch
    outputs = model(inputs)

    # Detach outputs and move to cpu
    outputs = [output.detach().cpu() for output in outputs]
    boxes = outputs[0]
    scores = outputs[1]
    labels = outputs[2]

    return boxes, scores, labels
