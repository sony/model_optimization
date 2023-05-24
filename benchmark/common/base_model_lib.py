from benchmark.common.sources import supported_datasets


class BaseModelLib(object):
    def __init__(self, args):
        self.args = args
        self.representative_dataset_folder = args.representative_dataset_folder
        self.validation_dataset_folder = args.validation_dataset_folder
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name

        if not self.dataset_name in supported_datasets(args.model_library):
            raise Exception(f'Dataset {args.dataset_name} is not supported for models from {args.model_library}')

    # return the float model (for quantization)
    def select_model(self):
        return

    # return the representative dataset (for quantization)
    def get_representative_dataset(self):
        return

    # perform evaluation for a given model
    def evaluate(self, model):
        return





