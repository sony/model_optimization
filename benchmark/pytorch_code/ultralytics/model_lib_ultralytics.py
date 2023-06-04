"""
 Parts of this file were copied from https://github.com/ultralytics/ultralytics and modified for this project needs.

 The Licence of the ultralytics project is shown in: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
"""


import torch
from torch.utils.data import DataLoader

from benchmark.pytorch_code.ultralytics.replacers import C2fModuleReplacer, DetectModuleReplacer, YOLOReplacer, DetectionModelModuleReplacer
from torchvision.transforms import transforms

from benchmark.pytorch_code.ultralytics.replacers import prepare_model_for_ultralytics_val
from benchmark.pytorch_code.helpers import get_representative_dataset
from ultralytics.yolo.data.dataset import YOLODataset

from benchmark.common.base_classes import BaseModelLib
from enum import Enum
from ultralytics.yolo.utils.torch_utils import initialize_weights


class ModelLib(BaseModelLib):

    def __init__(self, args):
        # Load model from ultralytics
        self.ultralytics_model = YOLOReplacer(args['model_name'])
        model_weights = self.ultralytics_model.model.state_dict()

        # replace few modules with quantization-friendly modules
        self.model = self.ultralytics_model.model
        self.model = DetectionModelModuleReplacer().replace(self.model)
        self.model = C2fModuleReplacer().replace(self.model)
        self.model = DetectModuleReplacer().replace(self.model)

        # load pre-trained weights
        initialize_weights(self.model)
        self.model.load_state_dict(model_weights)  # load weights
        super().__init__(args)

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
        stride = 32
        names = self.ultralytics_model.model.names

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

        return get_representative_dataset(dl, n_iter, 'img', transforms.Normalize(0,255))

    def evaluate(self, model):

        # Use Cuda device if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Some attributes are required for the evaluation of the quantized model
        self.ultralytics_model = prepare_model_for_ultralytics_val(self.ultralytics_model, model)

        # Evaluation using ultralytics interface
        results = self.ultralytics_model.val(batch=int(self.args['batch_size']))  # evaluate model performance on the validation set
        return results.mean_results()[-1]



