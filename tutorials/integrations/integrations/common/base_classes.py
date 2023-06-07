from abc import abstractmethod, ABC


class BaseModelLib(object):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    # return the float model (for quantization)
    def get_model(self):
        return

    @abstractmethod
    # return the representative dataset (for quantization)
    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size):
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


