import timm
from timm.data import create_dataset, create_loader, resolve_data_config

from integrations.common.base_classes import BaseModelLib
from integrations.pytorch.helpers import classification_eval, get_representative_dataset
from integrations.integrations.common.consts import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT

import logging



class ModelLib(BaseModelLib):

    def __init__(self, args):
        model_list = timm.list_models('')
        if args[MODEL_NAME] not in model_list:
            logging.error(f'Unknown model, Available timm models : {model_list}')
        self.model = timm.create_model(args[MODEL_NAME], pretrained=True)
        self.data_config = resolve_data_config([], model=self.model)  # include the pre-processing
        super().__init__(args)

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        train_dataset = create_dataset(name='ImageNet', root=representative_dataset_folder, split='train',
                                       is_training=False, batch_size=batch_size)
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
        batch_size = int(self.args[BATCH_SIZE])
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
        return classification_eval(model, testloader, self.args[VALIDATION_SET_LIMIT])



