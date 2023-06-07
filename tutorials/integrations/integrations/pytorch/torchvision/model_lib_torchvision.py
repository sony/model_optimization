import torch
import torchvision
from torch.utils.data import Subset
from torchvision import models

from integrations.common.base_classes import BaseModelLib
from integrations.pytorch.helpers import classification_eval, get_representative_dataset
from integrations.common.consts import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT


class ModelLib(BaseModelLib):

    @staticmethod
    def get_torchvision_model(model_name):
        # todo: replace with dedicated API (models_list(), get_model()...) when updating to torchvision 0.14
        return getattr(models, model_name)

    @staticmethod
    def get_torchvision_weights(model_name):
        # todo: replace with dedicated API (models_list(), get_model()...) when updating to torchvision 0.14
        return models.get_weight(model_name.title().replace('net', 'Net').replace('nas', 'NAS').replace('Mf', 'MF') + '_Weights.DEFAULT')

    def __init__(self, args):
        self.model = self.get_torchvision_model(args[MODEL_NAME])
        self.model = self.model(weights='DEFAULT')
        self.preprocess = self.get_torchvision_weights(args[MODEL_NAME]).transforms()
        super().__init__(args)

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        ds = torchvision.datasets.ImageNet(representative_dataset_folder, split='train', transform=self.preprocess)
        # ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, n_images)))
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)
        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        batch_size = int(self.args[BATCH_SIZE])
        testset = torchvision.datasets.ImageNet(self.validation_dataset_folder, split='val', transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return classification_eval(model, testloader, self.args[VALIDATION_SET_LIMIT])


