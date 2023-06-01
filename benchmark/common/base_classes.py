from abc import abstractmethod, ABC


class BaseModelLib(object):
    def __init__(self, args):
        self.args = args
        self.representative_dataset_folder = args['representative_dataset_folder']
        self.validation_dataset_folder = args['validation_dataset_folder']
        # self.dataset_name = args['dataset_name']
        self.model_name = args['model_name']
        model_library = args['model_library']

        # if not self.dataset_name in supported_datasets(model_library):
        #     raise Exception(f'Dataset {self.dataset_name} is not supported for models from {model_library}')

    @abstractmethod
    # return the float model (for quantization)
    def get_model(self):
        return

    @abstractmethod
    # return the representative dataset (for quantization)
    def get_representative_dataset(self):
        return

    @abstractmethod
    # perform evaluation for a given model
    def evaluate(self, model):
        return


class ModuleReplacer(ABC):

    def __init__(self):
        return

    def get_new_module(self):
        return

    def get_config(self):
        return

    def replace(self):
        return


