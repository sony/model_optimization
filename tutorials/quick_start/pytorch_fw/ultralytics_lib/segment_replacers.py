import torch
from overrides import override
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules import Segment
from ultralytics.utils.tal import make_anchors, dist2bbox

from tutorials.quick_start.common.model_lib import ModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.detect_replacers import DetectReplacer
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

    @override
    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        y_bb, y_cls = self.detect(self, x)

        if self.training:
            return (y_bb, y_cls), mc, p
        return y_bb, y_cls, mc, p


class SegmentModuleReplacer(ModuleReplacer):
    """
    A module replacer for Segment modules.
    """

    def get_new_module(self, config):
        return SegmentReplacer(*config)

    def get_config(self, c):
        nc = c.nc
        nm = c.nm
        npr = c.npr
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, nm, npr, ch]

    def replace(self, model):
        return replace_2d_deg_module(model, Segment, self.get_new_module, self.get_config)


class SegmentationValidatorReplacer(SegmentationValidator):
    """
    Replaces the DetectionValidator to include missing functionality from the Detect module.
    """
    @override
    def postprocess(self, preds):
        # Post-processing additional part - exported from Detect module
        stride = torch.tensor([8, 16, 32], dtype=torch.float32)
        grid = (self.args.imgsz / stride).numpy().astype(int)
        in_ch = 64 + self.nc  # 144
        x_dummy = [torch.ones(1, in_ch, grid[0], grid[0]), torch.ones(1, in_ch, grid[1], grid[1]),
                   torch.ones(1, in_ch, grid[2], grid[2])]
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(x_dummy, stride, 0.5))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        a = anchors.to(device)
        s = strides.to(device)
        y_bb, y_cls, masks_coeffs, proto = preds
        dbox = dist2bbox(y_bb, a.unsqueeze(0), xywh=True, dim=1) * s
        y = torch.cat((dbox, y_cls), 1)
        # additional part for segmentation
        preds = (torch.cat([y, masks_coeffs], 1), (y_cls, masks_coeffs, proto))

        # Original post-processing part
        preds = super().postprocess(preds)

        return preds