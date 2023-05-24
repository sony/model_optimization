import torch

import timm
from timm.data import create_dataset, create_loader, resolve_data_config

from benchmark.libraries.base_model_lib import BaseModelLib
from logging import error

from benchmark.utils.helpers import classification_eval, get_representative_dataset


class ModelLib(BaseModelLib):

    def __init__(self, args):
        super().__init__(args)

    def select_model(self, model_name):
        model_list = timm.list_models('')
        if model_name not in model_list:
            error(f'Unknown model, Available timm models : {model_list}')
        self.model = timm.create_model(model_name, pretrained=True)
        self.data_config = resolve_data_config([], model=self.model)
        return self.model.cuda()

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
        if self.dataset_name == 'IMAGENET':
            train_dataset = create_dataset(name='ImageNet', root=representative_dataset_folder, is_training=False,
                                         batch_size=batch_size)
        dl = create_loader(
            train_dataset,
            input_size=self.data_config['input_size'],
            batch_size=batch_size,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            crop_pct=self.data_config['crop_pct'])
        return get_representative_dataset(dl, n_iter)

    def evaluate(self, model):
        batch_size = self.args.batch_size
        val_dataset = create_dataset(name='', root=self.validation_dataset_folder, is_training=False,
                                           batch_size=batch_size)
        testloader = create_loader(
            val_dataset,
            input_size=self.data_config['input_size'],
            batch_size=batch_size,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            crop_pct=self.data_config['crop_pct'])
        return classification_eval(model, testloader)



