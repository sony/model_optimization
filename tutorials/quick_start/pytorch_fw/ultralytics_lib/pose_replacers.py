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

"""
 Parts of this file were copied from https://github.com/ultralytics/ultralytics and modified for this project needs.

 The Licence of the ultralytics project is shown in: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
"""
from typing import Tuple

import torch
from overrides import override
from torch import Tensor
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.nn.modules import Pose
from ultralytics.utils.tal import make_anchors, dist2bbox

from tutorials.quick_start.common.model_lib import ModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.detect_replacers import DetectReplacer, DetectionValidatorReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.replace_module import replace_2d_deg_module


class PoseReplacer(Pose):
    """
    Replaces the Pose module to use the replaced Detect forward function.
    To improve quantization (due to different data types), we removes the output concatenation.
    This will be added back in post_process.
    Also removes the key points decoding part, which will be added back in post_process.
    """

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, kpt_shape, ch)
        self.detect = DetectReplacer.forward

    @override
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        y_bb, y_cls = self.detect(self, x)
        return y_bb, y_cls, kpt


class PoseModuleReplacer(ModuleReplacer):
    """
    A module replacer for Segment modules.
    """

    def get_new_module(self, config: list) -> PoseReplacer:
        return PoseReplacer(*config)

    def get_config(self, c: torch.nn.Module) -> list:
        nc = c.nc
        kpt_shape = c.kpt_shape
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, kpt_shape, ch]

    def replace(self, model: torch.nn.Module) -> torch.nn.Module:
        return replace_2d_deg_module(model, Pose, self.get_new_module, self.get_config)


class PoseValidatorReplacer(PoseValidator, DetectionValidatorReplacer):
    """
    Replaces the PoseValidator to include missing functionality from the Pose module.
    Uses the post process function from the DetectionValidatorReplacer, and adds the keypoint post-processing.
    """
    @override
    def postprocess(self, preds: Tuple[Tensor, Tensor, Tensor]) -> list[Tensor]:
        kpt_shape = (17, 3)
        a, s = self.create_anchors_strides()

        y_bb, y_cls, kpts = preds
        dbox = dist2bbox(y_bb, a.unsqueeze(0), xywh=True, dim=1) * s
        detect_out = torch.cat((dbox, y_cls), 1)
        # additional part for pose estimation
        ndim = kpt_shape[1]
        pred_kpt = kpts.clone()
        if ndim == 3:
            pred_kpt[:, 2::3] = pred_kpt[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
        pred_kpt[:, 0::ndim] = (pred_kpt[:, 0::ndim] * 2.0 + (a[0] - 0.5)) * s
        pred_kpt[:, 1::ndim] = (pred_kpt[:, 1::ndim] * 2.0 + (a[1] - 0.5)) * s
        preds = (torch.cat([detect_out, pred_kpt], 1), (y_cls, kpts))

        # Original post-processing part
        preds = super().postprocess(preds)

        return preds
    