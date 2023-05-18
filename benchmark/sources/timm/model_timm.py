from logging import error
import torch


from model import Model

import timm
from timm.data import create_dataset, create_loader, resolve_data_config


class ModelTimm(Model):

    def __init__(self, args):

        model_list = timm.list_models('')
        if args.model_name not in model_list:
            error(f'Unknown model, Available timm models : {model_list}')

        self.model = timm.create_model(args.model_name, pretrained=True)
        self.data_config = resolve_data_config(vars(args), model=self.model)

        super().__init__(args)

    def get_model(self):
        return self.model.cuda()

    # return the representative dataset (for quantization)
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

        class RepresentativeDataset(object):
            def __init__(self, in_data_loader):
                self.dl = in_data_loader
                self.iter = iter(self.dl)

            def __call__(self):
                for _ in range(n_iter):
                    try:
                        x = next(self.iter)[0]
                    except StopIteration:
                        self.iter = iter(self.dl)
                        x = next(self.iter)[0]
                    # yield [torch.permute(x, [0, 2, 3, 1]).cpu().numpy()]
                    yield [x.cpu().numpy()]

        return RepresentativeDataset(dl)

    # perform evaluation for a given model
    def evaluation(self, model, args):
        batch_size = args.batch_size

        if self.dataset_name == 'IMAGENET':
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

        correct = 0
        total = 0
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = model(images.cuda())
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.cuda()).sum().item()
                if total % 100 == 0:
                    print(f'Num of images: {total}, Accuracy: {100 * correct // total} %')

        print(f'Num of images: {total}, Accuracy: {100 * correct // total} %')

        return correct // total



