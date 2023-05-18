import torch
from torch.utils.data import DataLoader

from sources.ultralytics.replacers import ReplacerYoloC2f, ReplacerYoloDetect, YOLO_modified, ReplacerYoloDetectionModel
from ultralytics.yolo.data.dataset import YOLODataset

from model import Model
from enum import Enum
from ultralytics.models import v8
from ultralytics.yolo.utils.torch_utils import initialize_weights


class ModelYolo(Model):

    def __init__(self, args):
        
        # get the model from ultralytics
        YAML_PATH = v8.__path__._path[0] + "/" + args.model_name + ".yaml"
        # self.ultralytics_model = YOLO(YAML_PATH)
        self.ultralytics_model = YOLO_modified('yolov8n.pt')

        # replace a few modules with quantization-friendly modules
        self.model = ReplacerYoloDetectionModel().replace(self.ultralytics_model.model)
        net = self.model
        net = ReplacerYoloC2f().replace(net)
        net = ReplacerYoloDetect().replace(net)

        # load pre-trained weights
        WEIGHTS_PATH = "/data/projects/swat/network_database/ModelZoo/Float-Pytorch-Models/yolov8/" + args.model_name + ".pt"
        initialize_weights(net)
        net.load_state_dict(torch.load(WEIGHTS_PATH))  # load weights

        # self.ultralytics_model = YOLO('yolov8n.pt')
        super().__init__(args)

    def get_model(self):
        return self.model.cuda()

    # return the representative dataset (for quantization)
    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
        stride = 32
        names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

        class hyp(Enum):
            mask_ratio = 4
            overlap_mask = True

        dataset = YOLODataset(
            img_path=representative_dataset_folder,
            imgsz=image_size,
            batch_size=batch_size,
            augment=False,  # augmentation
            hyp=hyp,  # TODO: probably add a get_hyps_from_cfg function
            rect=False,  # rectangular batches
            cache=None,
            single_cls=False,
            stride=int(stride),
            pad=0.5,
            prefix='',#colorstr(f'{mode}: '),
            use_segments=False,
            use_keypoints=False,
            names=names)

        dl = DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=False)
                      #num_workers=4)

        class RepresentativeDataset(object):
            def __init__(self, in_data_loader):
                self.dl = in_data_loader
                self.iter = iter(self.dl)

            def __call__(self):
                for _ in range(n_iter):
                    try:
                        x = next(self.iter)['img']
                    except StopIteration:
                        self.iter = iter(self.dl)
                        x = next(self.iter)['img']
                    # yield [torch.permute(x, [0, 2, 3, 1]).cpu().numpy()]
                    x = x/255
                    yield [x.cpu().numpy()]

        return RepresentativeDataset(dl)

    # perform evaluation for a given model
    def evaluation(self, model, batch_size):

        def fuse():
            return self.ultralytics_model.model

        setattr(model, 'args', self.ultralytics_model.model.args)
        setattr(model, 'fuse', fuse)
        setattr(model, 'names', self.ultralytics_model.model.names)
        setattr(model, 'stride', self.ultralytics_model.model.stride)
        setattr(self.ultralytics_model, 'model', model)

        return self.ultralytics_model.val()  # evaluate model performance on the validation set



