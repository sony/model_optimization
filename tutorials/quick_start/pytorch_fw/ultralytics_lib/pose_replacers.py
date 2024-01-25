import torch
from overrides import override
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.nn.modules import Pose
from ultralytics.utils.tal import make_anchors, dist2bbox

from tutorials.quick_start.common.model_lib import ModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.detect_replacers import DetectReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.replace_module import replace_2d_deg_module


class PoseReplacer(Pose):
    """
    Replaces the Pose module to use the replaced Detect forward function.
    To improve quantization (due to different data types), we removes the output concatenation.
    This will be added back in post_process.
    """

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        super().__init__(nc, kpt_shape, ch)
        self.detect = DetectReplacer.forward

    @override
    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        y_bb, y_cls = self.detect(self, x)
        if self.training:
            return (y_bb, y_cls), kpt
        return y_bb, y_cls, kpt


class PoseModuleReplacer(ModuleReplacer):
    """
    A module replacer for Segment modules.
    """

    def get_new_module(self, config):
        return PoseReplacer(*config)

    def get_config(self, c):
        nc = c.nc
        kpt_shape = c.kpt_shape
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, kpt_shape, ch]

    def replace(self, model):
        return replace_2d_deg_module(model, Pose, self.get_new_module, self.get_config)


class PoseValidatorReplacer(PoseValidator):
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
        kpt_shape = (17, 3)
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