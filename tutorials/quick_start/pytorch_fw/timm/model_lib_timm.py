import timm
from timm.data import create_dataset, create_loader, resolve_data_config

from common.model_lib import BaseModelLib
from pytorch_fw.utils import classification_eval, get_representative_dataset
from common.constants import MODEL_NAME, BATCH_SIZE, VALIDATION_SET_LIMIT, VALIDATION_DATASET_FOLDER, \
    IMAGENET_DATASET
from common.results import DatasetInfo


class ModelLib(BaseModelLib):

    def __init__(self, args):
        avialable_models = timm.list_models('')
        model_name = args[MODEL_NAME]
        if model_name in avialable_models:
            self.model = timm.create_model(args[MODEL_NAME], pretrained=True)
            self.data_config = resolve_data_config([], model=self.model)  # include the pre-processing
            self.dataset_name = IMAGENET_DATASET
            super().__init__(args)
        else:
            raise Exception(f'Unknown model name {model_name}, Available timm models : {avialable_models}')

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
        train_dataset = create_dataset(name='', root=representative_dataset_folder,
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
        validation_dataset_folder = self.args[VALIDATION_DATASET_FOLDER]
        val_dataset = create_dataset(name='', root=validation_dataset_folder, is_training=False,
                                           batch_size=batch_size)
        testloader = create_loader(
            val_dataset,
            input_size=self.data_config['input_size'],
            batch_size=batch_size,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            crop_pct=self.data_config['crop_pct'])

        acc, total = classification_eval(model, testloader, self.args[VALIDATION_SET_LIMIT])
        dataset_info = DatasetInfo(self.dataset_name, total)
        return acc, dataset_info



