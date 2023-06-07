"""
 Parts of this file were copied from https://github.com/ultralytics/ultralytics and modified for this project needs.

 The Licence of the ultralytics project is shown in: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
"""

from enum import Enum

import torch
from torch.utils.data import DataLoader

from integrations.pytorch.ultralytics.replacers import C2fModuleReplacer, DetectModuleReplacer, YOLOReplacer, DetectionModelModuleReplacer
from torchvision.transforms import transforms

from integrations.pytorch.ultralytics.replacers import prepare_model_for_ultralytics_val
from integrations.pytorch.helpers import get_representative_dataset
from integrations.common.base_classes import BaseModelLib
from integrations.integrations.common.consts import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT

from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.utils.torch_utils import initialize_weights


class ModelLib(BaseModelLib):

    def __init__(self, args):
        # Load model from ultralytics
        self.ultralytics_model = YOLOReplacer(args[MODEL_NAME])
        model_weights = self.ultralytics_model.model.state_dict()

        # Replace few modules with quantization-friendly modules
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

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        stride = 32
        names = self.ultralytics_model.model.names

        class hyp(Enum):
            mask_ratio = 4
            overlap_mask = True

        dataset = YOLODataset(
            img_path=representative_dataset_folder,
            imgsz=640,
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
        results = self.ultralytics_model.val(batch=int(self.args[BATCH_SIZE]))  # evaluate model performance on the validation set
        return results.mean_results()[-1]



