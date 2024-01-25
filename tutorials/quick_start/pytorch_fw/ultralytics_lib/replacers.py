# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

import torch

from common.model_lib import ModuleReplacer
from overrides import override
from ultralytics import YOLO
from ultralytics.nn.modules import C2f
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.checks import check_imgsz
from pathlib import Path

from model_compression_toolkit.core.pytorch.back2framework.pytorch_model_builder import PytorchModel
from tutorials.quick_start.pytorch_fw.ultralytics_lib.detect_replacers import DetectionValidatorReplacer, \
    DetectModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.pose_replacers import PoseValidatorReplacer, PoseModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.replace_module import replace_2d_deg_module
from tutorials.quick_start.pytorch_fw.ultralytics_lib.segment_replacers import SegmentationValidatorReplacer, \
    SegmentModuleReplacer

TASK_MAP = {
    'detect': {
        'moduleReplacer': DetectModuleReplacer(),
        'validator': DetectionValidatorReplacer},
    'segment': {
        'moduleReplacer': SegmentModuleReplacer(),
        'validator': SegmentationValidatorReplacer},
    'pose': {
        'moduleReplacer': PoseModuleReplacer(),
        'validator': PoseValidatorReplacer}
}


# In this section we slightly modify C2f module and replace the "list" function which is not supported by torch.fx
class C2fReplacer(C2f):
    """
    A new C2f module definition supported by torch.fx
    """
    @override
    def forward(self, x):
        y1 = self.cv1(x).chunk(2, 1)
        y = [y1[0], y1[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fModuleReplacer(ModuleReplacer):
    """
    A module replacer for C2f modules.
    """

    def get_new_module(self, config):
        return C2fReplacer(*config)

    def get_config(self, c):
        c1 = next(c.cv1.children()).in_channels
        c2 = next(c.cv1.children()).out_channels
        cc = c.c
        n = int(next(c.cv2.children()).in_channels / cc - 2)
        e = cc / c2
        g = next(next(next(c.m.children()).children()).children()).groups
        shortcut = next(c.m.children()).add
        return [c1, c2, n, shortcut, g, e]

    def replace(self, model):
        return replace_2d_deg_module(model, C2f, self.get_new_module, self.get_config)


class YOLOReplacer(YOLO):
    """
    Replaces the YOLO class to include the modified DetectionValidator
    """
    @override
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = False  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        overrides['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        args.data = "coco-pose.yaml" if self.task == 'pose' else args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task]['validator'](args=args)
        validator.model = self.model
        validator(model=self.model)
        self.metrics_data = validator.metrics

        return validator.metrics


def prepare_model_for_ultralytics_val(ultralytics_model, model, quantized=False):
    """
    Prepares the model for Ultralytics validation by setting necessary attributes.
    """
    class CustomPytorchModel(PytorchModel):
        def __init__(self, graph, append2output, return_float_outputs, wrapper, get_activation_quantizer_holder_fn):
            super().__init__(graph, append2output, return_float_outputs, wrapper, get_activation_quantizer_holder_fn)
            self.args = ultralytics_model.model.args
            self.names = ultralytics_model.model.names
            self.stride = ultralytics_model.model.stride
            self.device = next(model.parameters()).device
        def forward(self, x, augment=False, visualize=False, embed=False):
            return super().forward(x)

        def fuse(self, verbose=True):
            return self

    if not hasattr(model, 'args'):
        setattr(model, 'args', ultralytics_model.model.args)
        setattr(model, 'names', ultralytics_model.model.names)
        setattr(model, 'stride', ultralytics_model.model.stride)
        setattr(model, "device", next(model.parameters()).device)
        if quantized:  # cast to custom pytorch model that accepts augment, visualize and embed
            custom_model = CustomPytorchModel(graph=model.graph, append2output=model.append2output,
                                              return_float_outputs=model.return_float_outputs, wrapper=model.wrapper,
                                              get_activation_quantizer_holder_fn=model.get_activation_quantizer_holder)
            for attr_name, attr_value in vars(model).items():
                setattr(custom_model, attr_name, attr_value)

        setattr(ultralytics_model, 'model', custom_model if quantized else model)

    # ultralytics_model.to(next(model.parameters()).device)

    return ultralytics_model
