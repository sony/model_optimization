import timm
from timm.data import create_dataset, create_loader, resolve_data_config

from benchmark.common.base_model_lib import BaseModelLib
from logging import error

from benchmark.pytorch_code.helpers import classification_eval, get_representative_dataset


class ModelLib(BaseModelLib):

    def __init__(self, args):
        model_list = timm.list_models('')
        if args['model_name'] not in model_list:
            error(f'Unknown model, Available timm models : {model_list}')
        self.model = timm.create_model(args['model_name'], pretrained=True)
        self.data_config = resolve_data_config([], model=self.model)  # include the pre-processing
        super().__init__(args)

    def get_model(self):
        return self.model

    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):
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
        batch_size = int(self.args['batch_size'])
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
        return classification_eval(model, testloader, self.args['validation_set_limit'])



