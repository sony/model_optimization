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

# This tutorial demonstrates how to retrain and quantize MobileNetV2 on CIFAR100 using
# mixed-precision quantization. First we fine-tune a pretrained MobileNetV2 on ImageNet, then
# we use MCT post-training-quantization to compress the weights to 0.75 compression ratio.

import argparse
import copy
import tempfile
import random

import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from tqdm import tqdm

import model_compression_toolkit as mct
import numpy as np

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain_num_epochs', type=int, default=20,
                        help='Number of epochs for the retraining phase.')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Batch size for evaluation.')
    parser.add_argument('--retrain_batch_size', type=int, default=32,
                        help='Batch size for retraining.')
    parser.add_argument('--retrain_lr', type=float, default=0.001,
                        help='Learning rate to use during retraining.')
    parser.add_argument('--retrain_momentum', type=float, default=0.9,
                        help='SGD momentum to use during retraining.')

    parser.add_argument('--representative_dataset_dir', type=str, default='./data',
                        help='Folder path to save the representative dataset.')
    parser.add_argument('--ptq_batch_size', type=int, default=50,
                        help='Batch size for the representative data during PTQ calibration.')
    parser.add_argument('--num_calibration_iterations', type=int, default=10,
                        help='Number of iterations for calibration.')
    parser.add_argument('--weights_compression_ratio', type=float, default=0.75,
                        help='Weights compression ratio to use for mixed-precision quantization.')
    parser.add_argument('--mixed_precision_num_of_images', type=int, default=32,
                        help='Number of images to use for mixed-precision configuration search.')
    parser.add_argument('--enable_mixed_precision_gradients_weighting', action='store_true', default=False,
                        help='Whether to use gradients during mixed-precision configuration search or not.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed to set for randomness.')

    return parser.parse_args()



def get_cifar100_trainloader(dataset_folder, transform, train_batch_size):
    """
    Get CIFAR100 train loader.
    """
    trainset = torchvision.datasets.CIFAR100(root=dataset_folder, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)
    return trainloader


def get_cifar100_testloader(dataset_folder, transform, eval_batch_size):
    """
    Get CIFAR100 test loader.
    """
    testset = torchvision.datasets.CIFAR100(root=dataset_folder, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    return testloader


def evaluate(model, testloader, device):
    """
    Evaluate a model using testloader.

    Args:
        model: Model to evaluate.
        testloader: Test loader to use for evaluation.
        device: Device to run evaluation on.

    Returns:
        Valuation accuracy.

    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = (100 * correct / total)
    print('Accuracy: %.2f%%' % val_acc)
    return val_acc


def retrain(model, transform, device, args):
    trainloader = get_cifar100_trainloader(args.representative_dataset_dir,
                                           transform,
                                           args.retrain_batch_size)

    testloader = get_cifar100_testloader(args.representative_dataset_dir,
                                         transform,
                                         args.eval_batch_size)

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.retrain_lr,
                          momentum=args.retrain_momentum)

    best_acc = 0.0
    # Training loop
    for epoch in range(args.retrain_num_epochs):
        prog_bar = tqdm(enumerate(trainloader),
                        total=len(trainloader),
                        leave=True)

        print(f'Retrain epoch: {epoch}')
        for i, data in prog_bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, and update parameters
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, testloader, device)

        # Check if this model has the best accuracy, and if so, save it
        if val_acc > best_acc:
            print(f'Best accuracy so far {val_acc}')
            best_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    return model


if __name__ == '__main__':

    # Parse arguments
    args = argument_handler()

    seed_everything(args.seed)

    # Load pretrained MobileNetV2 model on ImageNet
    model = torchvision.models.mobilenet_v2(pretrained=True)

    # Modify last layer to match CIFAR-100 classes
    model.classifier[1] = nn.Linear(model.last_channel, 100)

    # Create preprocessing pipeline for training and evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to fit MobileNetV2 input
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # Normalize inputs to range [-1, 1]

    # If GPU available, move the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Fine-tune the model to adapt to CIFAR100
    model = retrain(model,
                    transform,
                    device,
                    args)

    # Evaluate the retrained model
    testloader = get_cifar100_testloader(args.representative_dataset_dir,
                                         transform,
                                         args.eval_batch_size)
    evaluate(model, testloader, device)

    # Create representative_data_gen function from the train dataset
    trainloader = get_cifar100_trainloader(args.representative_dataset_dir,
                                           transform,
                                           args.retrain_batch_size)


    def representative_data_gen() -> list:
        for _ in range(args.num_calibration_iterations):
            yield [next(iter(trainloader))[0]]


    # Get a TargetPlatformCapabilities object that models the hardware for the quantized model inference.
    # Here, for example, we use the default platform that is attached to a Pytorch layers representation.
    target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')

    # Create a mixed-precision quantization configuration with possible mixed-precision search options.
    # MCT will search a mixed-precision configuration (namely, bit-width for each layer)
    # and quantize the model according to this configuration.
    # The candidates bit-width for quantization should be defined in the target platform model:
    configuration = mct.core.CoreConfig(mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(
        num_of_images=args.mixed_precision_num_of_images,
        use_hessian_based_scores=args.enable_mixed_precision_gradients_weighting))

    # Get KPI information to constraint your model's memory size.
    # Retrieve a KPI object with helpful information of each KPI metric,
    # to constraint the quantized model to the desired memory size.
    kpi_data = mct.core.pytorch_kpi_data(model, representative_data_gen, configuration,
                                         target_platform_capabilities=target_platform_cap)

    # Set a constraint for each of the KPI metrics.
    # Create a KPI object to limit our returned model's size. Note that this values affects only layers and attributes
    # that should be quantized (for example, the kernel of Conv2D in Pytorch will be affected by this value,
    # while the bias will not)
    # examples:
    # weights_compression_ratio = 0.75 - About 0.75 of the model's weights memory size when quantized with 8 bits.
    kpi = mct.core.KPI(kpi_data.weights_memory * args.weights_compression_ratio)
    configuration.mixed_precision_config.set_target_kpi(kpi)

    # It is also possible to constraint only part of the KPI metric, e.g., by providing only weights_memory target
    # in the past KPI object, e.g., kpi = mct.core.KPI(kpi_data.weights_memory * 0.75)
    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(model,
                                                                                    representative_data_gen,
                                                                                    core_config=configuration,
                                                                                    target_platform_capabilities=target_platform_cap)
    # Finally, we evaluate the quantized model:
    print(f'Evaluating quantized model')
    evaluate(quantized_model,
             testloader,
             device)

    # Export quantized model to ONNX
    _, onnx_file_path = tempfile.mkstemp('.onnx')  # Path of exported model
    mct.exporter.pytorch_export_model(model=quantized_model,
                                      save_model_path=onnx_file_path,
                                      repr_dataset=representative_data_gen)
