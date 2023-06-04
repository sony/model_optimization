import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import models

from benchmark.common.base_model_lib import BaseModelLib

from benchmark.pytorch_code.helpers import classification_eval, get_representative_dataset


class ModelLib(BaseModelLib):

    @staticmethod
    def get_torchvision_model(model_name):
        # todo: replace with dedicated API (models_list(), get_mode()...) when updating to torchvision 0.14
        return getattr(models, model_name)

    @staticmethod
    def get_torchvision_weights(model_name):
        # todo: replace with dedicated API (models_list(), get_mode()...) when updating to torchvision 0.14
        return models.get_weight(model_name.title().replace('net', 'Net').replace('nas', 'NAS').replace('Mf', 'MF') + '_Weights.DEFAULT')

    def __init__(self, args):
        self.model = self.get_torchvision_model(args['model_name'])
        self.model = self.model(weights='DEFAULT')
        self.preprocess = self.get_torchvision_weights(args['model_name']).transforms()
        super().__init__(args)

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
        ds = torchvision.datasets.ImageNet(representative_dataset_folder, split='train', transform=self.preprocess)
        ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, n_images)))
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        batch_size = int(self.args['batch_size'])
        testset = torchvision.datasets.ImageNet(self.validation_dataset_folder, split='val', transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return classification_eval(model, testloader, self.args['validation_set_limit'])


