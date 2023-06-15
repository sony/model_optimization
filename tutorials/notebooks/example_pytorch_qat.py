# Copyright 2023 Sony Semiconductor Israel, Inc. All rights reserved.
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

"""
This tutorial demonstrates how to perform Quantization Aware Training (QAT) using the Model Compression Toolkit (MCT).
We first train a simple model on MNIST dataset, then quantize the model and transform it to a QAT-ready model.
A QAT-ready model is a model in which certain layers are wrapped by "quantization wrappers" with requested quantizers.
The user can now Fine-Tune the QAT-ready model. Finally, the model is finalized by the MCT which means the
MCT replaces the "quantization wrappers" with their native layers and quantized weights.
"""


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import model_compression_toolkit as mct
import tempfile


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--representative_dataset_dir', type=str, required=True, default=None,
                        help='folder path for the representative dataset.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for the representative data.')
    parser.add_argument('--num_calibration_iterations', type=int, default=10,
                        help='number of iterations for calibration.')
    return parser.parse_args()


# Let us define the network and some helper functions to train and evaluate the model.
# These are taken from the official Pytorch examples https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Evaluation function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    # Parse arguments
    args = argument_handler()

    # Set some training parameters
    batch_size = args.batch_size
    test_batch_size = 1000
    random_seed = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    dataset_folder = args.representative_dataset_dir
    epochs = 1
    gamma = 0.1
    lr = 1.0

    # Train a Pytorch model on MNIST
    # Let us define the dataset loaders, and optimizer and train the model for 2 epochs.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST(dataset_folder, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(dataset_folder, train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, num_workers=0, pin_memory=True, batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, num_workers=0, pin_memory=True, batch_size=test_batch_size,
                                              shuffle=False)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    # Hardware-friendly quantization using MCT
    def get_tpc():
        """
        Assuming a target hardware that uses power-of-2 thresholds and quantizes weights and activations
        to 2 and 3 bits, accordingly. Our assumed hardware does not require quantization of some layers
        (e.g. Flatten & Droupout).
        This function generates a TargetPlatformCapabilities with the above specification.

        Returns:
             TargetPlatformCapabilities object
        """
        tp = mct.target_platform
        default_config = tp.OpQuantizationConfig(
            activation_quantization_method=tp.QuantizationMethod.SYMMETRIC,
            weights_quantization_method=tp.QuantizationMethod.SYMMETRIC,
            activation_n_bits=3,
            weights_n_bits=2,
            weights_per_channel_threshold=True,
            enable_weights_quantization=True,
            enable_activation_quantization=True,
            quantization_preserving=False,
            fixed_scale=1.0,
            fixed_zero_point=0,
            weights_multiplier_nbits=0)

        default_configuration_options = tp.QuantizationConfigOptions([default_config])
        tp_model = tp.TargetPlatformModel(default_configuration_options)
        with tp_model:
            tp_model.set_quantization_format(quantization_format=tp.quantization_format.QuantizationFormat.FAKELY_QUANT)
            tp.OperatorsSet("NoQuantization",
                            tp.get_default_quantization_config_options().clone_and_edit(
                                enable_weights_quantization=False,
                                enable_activation_quantization=False))

        tpc = tp.TargetPlatformCapabilities(tp_model)
        with tpc:
            # No need to quantize Flatten and Dropout layers
            tp.OperationsSetToLayers("NoQuantization", [nn.Dropout,
                                                        torch.flatten])

        return tpc

    # Prepare a representative dataset callable from the MNIST training images for calibrating the initial
    # quantization parameters by the MCT.
    image_data_loader = iter(train_loader)
    def representative_data_gen():
        for _ in range(args.num_calibration_iterations):
            yield [next(image_data_loader)[0]]

    # Prepare model for QAT with MCT and return to user for fine-tuning. Due to the relatively easy
    # task of quantizing model trained on MNIST, we use a custom TPC in this example to demonstrate the degradation
    # caused by post training quantization
    qat_model, quantization_info = mct.qat.pytorch_quantization_aware_training_init(
        model,
        representative_data_gen,
        target_platform_capabilities=get_tpc())

    # Evaluate QAT-ready model accuracy from MCT. This model is fully quantized with "quantize wrappers"
    # accuracy is expected to be significantly lower
    test(qat_model, device, test_loader)

    # Fine-tune QAT model from MCT to recover the accuracy.
    optimizer = optim.Adam(qat_model.parameters(), lr=lr/10000)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epochs + 1):
        train(qat_model, device, train_loader, optimizer, epoch)
        test(qat_model, device, test_loader)
        scheduler.step()

    # Finalize QAT model: remove "quantize wrappers" and keep weights quantized as fake-quant values
    quantized_model = mct.qat.pytorch_quantization_aware_training_finalize(qat_model)

    # Re-evaluate accuracy after finalizing the model (should have a better accuracy than QAT model, since now the
    # activations are not quantized)
    test(quantized_model, device, test_loader)

    # Export quantized model to ONNX
    _, onnx_file_path = tempfile.mkstemp('.onnx') # Path of exported model
    mct.exporter.pytorch_export_model(model=quantized_model, save_model_path=onnx_file_path,
                                      repr_dataset=representative_data_gen, target_platform_capabilities=get_tpc(),
                                      serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX)