import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import models
import torchvision.transforms as transforms

from benchmark.libraries.base_model_lib import BaseModelLib

from benchmark.utils.helpers import classification_eval, get_representative_dataset


class ModelLib(BaseModelLib):

    def __init__(self, args):
        self.preprocess = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ])
        super().__init__(args)

    def select_model(self, model_name):
        self.model = getattr(models, model_name)
        return self.model(weights='IMAGENET1K_V1').cuda()

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
        ds = torchvision.datasets.ImageNet(representative_dataset_folder, split='val', transform=self.preprocess)
        ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, n_images)))
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        batch_size = self.args.batch_size
        testset = torchvision.datasets.ImageNet(self.validation_dataset_folder, split='val', transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return classification_eval(model, testloader)


