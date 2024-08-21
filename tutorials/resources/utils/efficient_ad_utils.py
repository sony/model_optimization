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

import os
import random
import shutil
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import roc_auc_score
import tifffile
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Global constants
IMAGE_SIZE = 256
OUT_CHANNELS = 384
SEED = 42

# Transform definitions
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TRANSFORM_AE = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

class ImageFolderWithoutTarget(datasets.ImageFolder):
    """Custom dataset that includes only images, no labels."""
    def __getitem__(self, index):
        sample, _ = super().__getitem__(index)
        return sample

class ImageFolderWithPath(datasets.ImageFolder):
    """Custom dataset that includes image paths along with images and labels."""
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def infinite_dataloader(loader):
    """Create an infinite dataloader that cycles through the dataset."""
    for data in itertools.cycle(loader):
        yield data

def train_transform(image):
    """Apply transformations to the training images."""
    return DEFAULT_TRANSFORM(image), DEFAULT_TRANSFORM(TRANSFORM_AE(image))

def visualize_anomalies(unified_model, dataset_path, test_output_dir=None, desc='Running inference'):
    """Visualize anomalies by overlaying heatmaps on the original images."""
    test_set = ImageFolderWithPath(os.path.join(dataset_path, 'bottle', 'test'))
    images_to_display = random.sample(list(test_set), 10)  # Randomly select 10 images to display
    y_true, y_score = [], []

    for image, target, path in tqdm(images_to_display, desc=desc):
        orig_width, orig_height = image.size
        image_tensor = DEFAULT_TRANSFORM(image)[None]  # Add batch dimension
        map_combined = unified_model(image_tensor)
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].detach().cpu().numpy()

        heatmap = cm.jet(map_combined)  # Apply colormap
        heatmap = np.uint8(cm.ScalarMappable(cmap='jet').to_rgba(map_combined) * 255)
        heatmap_pil = Image.fromarray(heatmap, 'RGBA').convert('RGB')  # Convert RGBA to RGB
        image_pil = image.convert('RGB')  # Ensure the original image is in RGB

        combined_image = Image.new('RGB', (orig_width * 2, orig_height))
        combined_image.paste(image_pil, (0, 0))
        combined_image.paste(heatmap_pil, (orig_width, 0))

        defect_class = os.path.basename(os.path.dirname(path))
        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        prediction_correct = 'Correct' if (y_score_image > 0.5) == y_true_image else 'Incorrect'
        defect_status = 'Defect' if y_true_image == 1 else 'No Defect'

        plt.figure(figsize=(12, 6))
        plt.imshow(combined_image)
        plt.title(f"Actual: {defect_status}, Prediction: {prediction_correct}")
        plt.axis('off')
        plt.show()

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

def train_ad(train_steps, dataset_path, sub_dataset, autoencoder, teacher, student):
    """Train the anomaly detection model."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_output_dir = os.path.join('output', 'trainings', 'mvtec_ad', sub_dataset)
    test_output_dir = os.path.join('output', 'anomaly_maps', 'mvtec_ad', sub_dataset, 'test')
    shutil.rmtree(train_output_dir, ignore_errors=True)
    shutil.rmtree(test_output_dir, ignore_errors=True)
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    full_train_set = ImageFolderWithoutTarget(os.path.join(dataset_path, sub_dataset, 'train'), transform=transforms.Lambda(train_transform))
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1)
    train_loader_infinite = infinite_dataloader(train_loader)

    teacher.eval()
    student.train()
    autoencoder.train()

    if torch.cuda.is_available():
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    optimizer = torch.optim.Adam(itertools.chain(student.parameters(), autoencoder.parameters()), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * train_steps), gamma=0.1)

    for iteration, (image_st, image_ae) in enumerate(train_loader_infinite, start=1):
        if torch.cuda.is_available():
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()

        student_output_st = student(image_st)[:, :OUT_CHANNELS]
        loss_st = torch.mean((teacher(image_st) - student_output_st) ** 2)

        ae_output = autoencoder(image_ae)
        loss_ae = torch.mean((teacher(image_ae) - ae_output) ** 2)

        loss_total = loss_st + loss_ae
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm.write(f"Step {iteration}, Loss: {loss_total.item():.4f}")

        if iteration >= train_steps:
            break

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir, 'autoencoder_final.pth'))

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    """Predict using the trained models and calculate anomaly maps."""
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :OUT_CHANNELS])**2, dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output - student_output[:, OUT_CHANNELS:])**2, dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def predict_combined(image, unified_model, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    """Predict using the trained models and calculate anomaly maps."""
    map_st, map_ae = unified_model(image)
    map_st = torch.mean(map_st, dim=1, keepdim=True)
    map_ae = torch.mean(map_ae, dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder, teacher_mean, teacher_std, desc='Map normalization'):
    """Normalize the anomaly maps generated by the models."""
    maps_st, maps_ae = [], []
    for image, _ in tqdm(validation_loader, desc=desc):
        if torch.cuda.is_available():
            image = image.cuda()
        map_combined, map_st, map_ae = predict(image, teacher, student, autoencoder, teacher_mean, teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, 0.9)
    q_st_end = torch.quantile(maps_st, 0.995)
    q_ae_start = torch.quantile(maps_ae, 0.9)
    q_ae_end = torch.quantile(maps_ae, 0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    """Calculate the normalization parameters for the teacher model outputs."""
    mean_outputs, mean_distances = [], []
    for train_image, _ in tqdm(train_loader, desc='Computing normalization parameters'):
        if torch.cuda.is_available():
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
        distance = (teacher_output - mean_output[None, :, None, None]) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)[None, :, None, None]
    channel_std = torch.sqrt(torch.mean(torch.stack(mean_distances), dim=0))[None, :, None, None]
    return channel_mean, channel_std