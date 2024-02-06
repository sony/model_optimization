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
from typing import Tuple, Any

import torch
from torch import Tensor
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules import Segment
from ultralytics.utils.tal import make_anchors, dist2bbox

from tutorials.quick_start.common.model_lib import ModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.detect_replacers import DetectReplacer, DetectionValidatorReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.replace_module import replace_2d_deg_module


class SegmentReplacer(Segment):
    """
    Replaces the Segment module to use the replaced Detect forward function.
    To improve quantization (due to different data types), we removes the output concatenation.
    This will be added back in post_process.
    """

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        super().__init__(nc, nm, npr, ch)
        self.detect = DetectReplacer.forward

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Any]:
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        y_bb, y_cls = self.detect(self, x)

        return y_bb, y_cls, mc, p


class SegmentModuleReplacer(ModuleReplacer):
    """
    A module replacer for Segment modules.
    """

    def get_new_module(self, config: list) -> SegmentReplacer:
        return SegmentReplacer(*config)

    def get_config(self, c: torch.nn.Module) -> list[int, int, int, tuple]:
        nc = c.nc
        nm = c.nm
        npr = c.npr
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, nm, npr, ch]

    def replace(self, model: torch.nn.Module) -> torch.nn.Module:
        return replace_2d_deg_module(model, Segment, self.get_new_module, self.get_config)


class SegmentationValidatorReplacer(SegmentationValidator, DetectionValidatorReplacer):
    """
    Replaces the SegmentationValidator to include missing functionality from the Segment module.
    Uses the post process function from the DetectionValidatorReplacer, and adds the segmentation post-processing.
    """
    def postprocess(self, preds: Tuple[Tensor, Tensor, Tensor, Any]) -> Tuple[list[Tensor], Tensor]:
        a, s = self.create_anchors_strides()
        y_bb, y_cls, masks_coeffs, proto = preds
        dbox = dist2bbox(y_bb, a.unsqueeze(0), xywh=True, dim=1) * s
        y = torch.cat((dbox, y_cls), 1)
        # additional part for segmentation
        preds = (torch.cat([y, masks_coeffs], 1), (y_cls, masks_coeffs, proto))

        # Original post-processing part
        preds = super().postprocess(preds)

        return preds