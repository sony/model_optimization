from logging import error

from benchmark.quant import quant
from benchmark.sources.sources import supported_datasets


class Model(object):
    def __init__(self, args):
        self.representative_dataset_folder = args.representative_dataset_folder
        self.validation_dataset_folder = args.validation_dataset_folder
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name

        if not self.dataset_name in supported_datasets(args.model_source):
            raise Exception(f'Dataset {args.dataset_name} is not supported for models from {args.model_source}')

    # return the float model (for quantization)
    def get_model(self):
        return

    # return the representative dataset (for quantization)
    def get_representative_dataset(self):
        return

    def quantize(self, args):
        return quant(self, args)

    # perform evaluation for a given model
    def evaluation(self, model):
        return





