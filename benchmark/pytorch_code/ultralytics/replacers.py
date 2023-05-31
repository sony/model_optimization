import torch

from benchmark.common.module_replacer import ModuleReplacer
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.v8.detect import DetectionValidator
from pathlib import Path
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors


def replace_2d_deg_module(model, old_module, new_module, get_config):
    for n, m in model.named_children():
        for name, c in m.named_children():
            if isinstance(c, old_module):
                l = new_module(get_config(c))
                setattr(l, 'f', c.f)
                setattr(l, 'i', c.i)
                setattr(l, 'type', c.type)
                setattr(m, name, l)
    return model


class C2fReplacer(C2f):

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y1 = self.cv1(x).chunk(2, 1)
        y = [y1[0], y1[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fModuleReplacer(ModuleReplacer):

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


class DetectReplacer(Detect):

    def forward(self, x):
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        # elif self.dynamic or self.shape != shape:
        #     self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        #     self.shape = shape

        if self.export and self.format == 'edgetpu':  # FlexSplitV ops issue
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
                (self.reg_max * 4, self.nc), 1)

        # idanb: move the last part of the detection head to post-processing (otherwise quantization is problematic)

        # dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        # y = torch.cat((dbox, cls.sigmoid()), 1)
        # return y if self.export else (y, x)
        y_cls = cls.sigmoid()
        y_bb = self.dfl(box)
        return (y_bb, y_cls)


class DetectModuleReplacer(ModuleReplacer):

    def get_new_module(self, config):
        return DetectReplacer(*config)

    def get_config(self, c):
        nc = c.nc
        ch = [next(next(x.children()).children()).in_channels for x in c.cv2.children()]
        return [nc, ch]

    def replace(self, model):
        return replace_2d_deg_module(model, Detect, self.get_new_module, self.get_config)


class DetectionModelReplacer(DetectionModel):
    def forward(self, x, augment=False, profile=False, visualize=False):
        # if augment:
        #     return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in
                                                         m.f]  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     LOGGER.info('visualize feature not yet supported')
        return x


class DetectionModelModuleReplacer(ModuleReplacer):

    def get_config(self, c):
        return [c.yaml]

    def get_new_module(self, config):
        return DetectionModelReplacer(*config)

    def replace(self, model):
        return self.get_new_module(self.get_config(model))


class DetectionValidatorReplacer(DetectionValidator):

    def postprocess(self, preds):

        # post-processing additional part - exported from the model
        x = [torch.ones(1, 144, 80, 80), torch.ones(1, 144, 40, 40), torch.ones(1, 144, 20, 20)]
        if preds[0].shape[2] == 6300:
            x = [torch.ones(1, 144, 80, 60), torch.ones(1, 144, 40, 30), torch.ones(1, 144, 20, 15)]
        anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, torch.Tensor([8,16,32]), 0.5))
        a = torch.Tensor.cuda(anchors)
        s = torch.Tensor.cuda(strides)
        dbox = dist2bbox(preds[0], a.unsqueeze(0), xywh=True, dim=1) * s
        preds = torch.cat((dbox, preds[1]), 1)

        preds = super().postprocess(preds)

        return preds


class YOLOReplacer(YOLO):

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
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = DetectionValidatorReplacer(args=args)
        validator(model=self.model)
        self.metrics_data = validator.metrics

        return validator.metrics


def prepare_model_for_ultralytics_val(ultralytics_model, model):

    if not hasattr(model, 'args'):
        def fuse():
            return ultralytics_model.model

        setattr(model, 'args', ultralytics_model.model.args)
        setattr(model, 'fuse', fuse)
        setattr(model, 'names', ultralytics_model.model.names)
        setattr(model, 'stride', ultralytics_model.model.stride)
        setattr(ultralytics_model, 'model', model)

    return ultralytics_model
