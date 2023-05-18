from logging import error

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from torchvision import models
import torchvision.transforms as transforms

from model import Model


class ModelTorchvision(Model):

    def __init__(self, args):

        self.model = getattr(models, args.model_name)

        if args.dataset_name == 'IMAGENET':
            # define the preprocessing
            self.preprocess = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])

        else:
            error('Unsupported dataset for this model')

        super().__init__(args)

    def get_model(self):
        # return models.mobilenet_v2(weights='IMAGENET1K_V1').cuda()
        return self.model(weights='IMAGENET1K_V1').cuda()

    # return the representative dataset (for quantization)
    def get_representative_dataset(self, representative_dataset_folder, n_iter, batch_size, n_images, image_size,
                                   preprocessing=None, seed=0):

        if self.dataset_name == 'IMAGENET':
            ds = torchvision.datasets.ImageNet(self.validation_dataset_folder, split='val', transform=self.preprocess)

        ds = Subset(ds, list(np.random.randint(0, len(ds) + 1, n_images)))
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, num_workers=4)

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
            testset = torchvision.datasets.ImageNet(self.validation_dataset_folder, split='val', transform=self.preprocess)
        testloader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

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



