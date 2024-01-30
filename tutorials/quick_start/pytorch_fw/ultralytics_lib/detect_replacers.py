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
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn import DetectionModel
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import make_anchors, dist2bbox

from tutorials.quick_start.common.constants import EDGE_TPU
from tutorials.quick_start.common.model_lib import ModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.replace_module import replace_2d_deg_module


# In this section we modify Detect module to exclude dynamic condition which is not supported by torch.fx
# In addition, we remove the last part of the detection head which is essential for improving the quantization
# This missing part will be added to the postprocessing implementation
class DetectReplacer(Detect):
    """
    Replaces the Detect module with modifications to support torch.fx and removes the last part of the detection head.
    """

    @override
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.export and self.format == EDGE_TPU:  # FlexSplitV ops issue
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
                (self.reg_max * 4, self.nc), 1)

        y_cls = cls.sigmoid()
        y_bb = self.dfl(box)
        return y_bb, y_cls


class DetectModuleReplacer(ModuleReplacer):
    """
    A module replacer for Detect modules.
    """

    def get_new_module(self, config: list) -> DetectReplacer:
        return DetectReplacer(*config)

    def get_config(self, c: torch.nn.Module) -> list[int, tuple]:
        nc = c.nc
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, ch]

    def replace(self, model: torch.nn.Module) -> torch.nn.Module:
        return replace_2d_deg_module(model, Detect, self.get_new_module, self.get_config)


# In this section we modify the DetectionModel to exclude dynamic condition which is not supported by torch.fx
class DetectionModelReplacer(DetectionModel):
    """
    Replaces the DetectionModel to exclude dynamic condition not supported by torch.fx.
    """

    # Original forward functions uses *args and **kwargs and these are not supported by torch.fx and therefore
    # we need to overload the forward function
    def forward(self, x, profile=False, visualize=False, augment=False, embed=False):
        return self.predict(x, profile=False, visualize=False, augment=False, embed=False)

    # Original predict uses a dynamic condition (if augment) which is not supported by torch.fx and we remove it
    @override
    def predict(self, x, profile=False, visualize=False, augment=False, embed=False):
        return self._predict_once(x, profile, visualize)  # single-scale inference, train

    # Original _predict_once uses a dynamic condition (if profile, if visualize, if embed) which are not supported by
    # torch.fx and we remove them
    @override
    def _predict_once(self, x, profile=False, visualize=False, embed=False):
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in
                                                         m.f]  # from earlier layers

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        return x


class DetectionModelModuleReplacer(ModuleReplacer):
    """
    A module replacer for DetectionModel modules.
    """

    def get_config(self, c: torch.nn.Module) -> list:
        return [c.yaml]

    def get_new_module(self, config: list) -> DetectionModelReplacer:
        return DetectionModelReplacer(*config)

    def replace(self, model: torch.nn.Module) -> torch.nn.Module:
        return self.get_new_module(self.get_config(model))


# In this section we modify the DetectionValidator (not part of the model) to include the missing functionality
# that was removed from the Detect module
class DetectionValidatorReplacer(DetectionValidator):
    """
    Replaces the DetectionValidator to include missing functionality from the Detect module.
    """

    # Code modified from make_anchors function in /ultralytics/utils/tal.py since we remove this part from the forward
    # function in the replaced Detect module
    def create_anchors_strides(self):
        # Post-processing additional part - exported from Detect module
        strides = [8, 16, 32]
        stride = torch.tensor(strides, dtype=torch.float32)
        grid = (self.args.imgsz / stride).numpy().astype(int)
        in_ch = 64 + self.nc  # 144
        x_dummy = [torch.ones(1, in_ch, grid[0], grid[0]), torch.ones(1, in_ch, grid[1], grid[1]),
                   torch.ones(1, in_ch, grid[2], grid[2])]
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(x_dummy, stride, 0.5))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = anchors.to(device)
        s = strides.to(device)
        return a, s

    @override
    def postprocess(self, preds: tuple[Tensor, Tensor]) -> list[Tensor]:
        a, s = self.create_anchors_strides()
        dbox = dist2bbox(preds[0], a.unsqueeze(0), xywh=True, dim=1) * s
        preds = torch.cat((dbox, preds[1]), 1)

        # Original post-processing part
        preds = super().postprocess(preds)

        return preds
