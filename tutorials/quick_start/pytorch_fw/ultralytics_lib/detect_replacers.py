# In this section we modify Detect module to exclude dynamic condition which is not supported by torch.fx
# In addition, we remove the last part of the detection head which is essential for improving the quantization
# This missing part will be added to the postprocessing implementation
import torch
from overrides import override
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn import DetectionModel
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import make_anchors, dist2bbox

from tutorials.quick_start.common.model_lib import ModuleReplacer
from tutorials.quick_start.pytorch_fw.ultralytics_lib.replace_module import replace_2d_deg_module


class DetectReplacer(Detect):
    """
    Replaces the Detect module with modifications to support torch.fx and removes the last part of the detection head.
    """
    @override
    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x

        if self.export and self.format == 'edgetpu':  # FlexSplitV ops issue
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
                (self.reg_max * 4, self.nc), 1)

        y_cls = cls.sigmoid()
        y_bb = self.dfl(box)
        return (y_bb, y_cls)


class DetectModuleReplacer(ModuleReplacer):
    """
    A module replacer for Detect modules.
    """

    def get_new_module(self, config):
        return DetectReplacer(*config)

    def get_config(self, c):
        nc = c.nc
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, ch]

    def replace(self, model):
        return replace_2d_deg_module(model, Detect, self.get_new_module, self.get_config)


# In this section we modify the DetectionModel to exclude dynamic condition which is not supported by torch.fx
class DetectionModelReplacer(DetectionModel):
    """
    Replaces the DetectionModel to exclude dynamic condition not supported by torch.fx.
    """
    def forward(self, x, profile=False, visualize=False, augment=False, embed=False):
        return self.predict(x, profile=False, visualize=False, augment=False, embed=False)

    @override
    def predict(self, x, profile=False, visualize=False, augment=False, embed=False):
        return self._predict_once(x, profile, visualize)  # single-scale inference, train

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

    def get_config(self, c):
        return [c.yaml]

    def get_new_module(self, config):
        return DetectionModelReplacer(*config)

    def replace(self, model):
        return self.get_new_module(self.get_config(model))

# In this section we modify the DetectionValidator (not part of tthe model) to include the missing functionality
# that was removed from the Detect module
class DetectionValidatorReplacer(DetectionValidator):
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
        dbox = dist2bbox(preds[0], a.unsqueeze(0), xywh=True, dim=1) * s
        preds = torch.cat((dbox, preds[1]), 1)

        # Original post-processing part
        preds = super().postprocess(preds)

        return preds
