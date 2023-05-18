
class Model(object):
    def __init__(self, args):
        self.representative_dataset_folder = args.train_data_path
        self.validation_dataset_folder = args.val_data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name

    # return the float model (for quantization)
    def get_model(self):
        pass

    # return the representative dataset (for quantization)
    def get_representative_dataset(self):
        pass

    # perform evaluation for a given model
    def evaluation(self, model):
        pass





