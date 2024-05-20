# Copyright 2024 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import logging
from tqdm import tqdm


def classification_eval(model, data_loader, limit=None, num_images_to_display=1000):
    logging.info(f'Start classification evaluation')
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Classification evaluation"):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images.to(device))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            if total % num_images_to_display == 0:
                print(f'Num of images: {total}, Accuracy: {round(100 * correct / total, 2)} %')
            if limit and total >= int(limit):
                break

    logging.info(f'Num of images: {total}, Accuracy: {round(100 * correct / total, 2)} %')

    return correct / total, total


def get_representative_dataset(data_loader, n_iters, data_loader_key=0, transforms=None):
    class RepresentativeDataset(object):
        def __init__(self, in_data_loader):
            self.dl = in_data_loader
            self.iter = iter(self.dl)

        def __call__(self):
            for _ in range(n_iters):
                try:
                    x = next(self.iter)[data_loader_key]
                except StopIteration:
                    self.iter = iter(self.dl)
                    x = next(self.iter)[data_loader_key]
                if transforms is not None:
                    x = transforms(x.float())
                yield [x.cpu().numpy()]

    return RepresentativeDataset(data_loader)
