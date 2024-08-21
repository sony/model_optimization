# The following code was mostly duplicated from https://github.com/nelson1425/EfficientAD
# and changed to generate an equivalent PyTorch model suitable for quantization.
# Main changes:
#   * Modify layers to make them more suitable for quantization.
#   * Inheritance class from HuggingFace
#   * Uninfied version of model combining the three subversions
# ==============================================================================
"""
Efficient Anomaly Detection Model - PyTorch implementation

This code contains a PyTorch implementation of efficient ad model, following
https://github.com/nelson1425/EfficientAD. This implementation includes a unified version of the model that combines the three submodels 
into one to ease the process of quantization and deployment. 

The code is organized as follows:
- 
- primary model definition - UnifiedAnomalyDetectionModel
- sub models 
- auto encoder - get_autoencoder
- student and teacher models - get_pdn_small

For more details on the model, refer to the original repository:
https://github.com/nelson1425/EfficientAD

"""
from torch import nn
from torchvision.datasets import ImageFolder
import torch
import json

def get_autoencoder(out_channels=384):
    """
    Constructs an autoencoder model with specified output channels.
    
    Parameters:
    - out_channels (int): The number of output channels in the final convolutional layer.
    
    Returns:
    - nn.Sequential: A PyTorch sequential model representing the autoencoder.
    """
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    """
    Constructs a small PDN (Pyramidal Decomposition Network) model.
    
    Parameters:
    - out_channels (int): The number of output channels in the final convolutional layer.
    - padding (bool): If True, applies padding to convolutional layers.
    
    Returns:
    - nn.Sequential: A PyTorch sequential model representing the small PDN.
    """
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )

class UnifiedAnomalyDetectionModel(nn.Module):
    """
    A unified model for anomaly detection combining teacher, student, and autoencoder models.
    
    Parameters:
    - teacher (nn.Module): The teacher model.
    - student (nn.Module): The student model.
    - autoencoder (nn.Module): The autoencoder model.
    - out_channels (int): Number of output channels in the student's output used for comparison.
    - teacher_mean (float): Mean used for normalizing the teacher's output.
    - teacher_std (float): Standard deviation used for normalizing the teacher's output.
    - q_st_start (float, optional): Start quantile for student-teacher comparison normalization.
    - q_st_end (float, optional): End quantile for student-teacher comparison normalization.
    - q_ae_start (float, optional): Start quantile for autoencoder-student comparison normalization.
    - q_ae_end (float, optional): End quantile for autoencoder-student comparison normalization.
    
    Methods:
    - forward(input_image): Processes the input image through the model.
    - save_model(filepath): Saves the model state to a file.
    - load_model(filepath, teacher_model, student_model, autoencoder_model): Loads the model state from a file.
    """
    def __init__(self, teacher, student, autoencoder, out_channels, teacher_mean, teacher_std, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
        super(UnifiedAnomalyDetectionModel, self).__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.out_channels = out_channels
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        self.q_st_start = q_st_start
        self.q_st_end = q_st_end
        self.q_ae_start = q_ae_start
        self.q_ae_end = q_ae_end

    def forward(self, input_image):
        teacher_output = self.teacher(input_image)
        student_output = self.student(input_image)
        autoencoder_output = self.autoencoder(input_image)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        student_output_st = student_output[:, :self.out_channels]
        student_output_ae = student_output[:, self.out_channels:]

        # Calculate MSE between teacher-student and autoencoder-student
        mse_st = (teacher_output - student_output_st) * (teacher_output - student_output_st)
        mse_ae = (autoencoder_output - student_output_ae) * (autoencoder_output - student_output_ae)

        return mse_st , mse_ae

    def save_model(self, filepath):
        """ Save the entire model including sub-models and parameters """
        model_info = {
            'model_state_dict': self.state_dict(),
            'out_channels': self.out_channels,
            'teacher_mean': self.teacher_mean.tolist(),
            'teacher_std': self.teacher_std.tolist(),
            'q_st_start': self.q_st_start.tolist() if self.q_st_start is not None else None,
            'q_st_end': self.q_st_end.tolist() if self.q_st_end is not None else None,
            'q_ae_start': self.q_ae_start.tolist() if self.q_ae_start is not None else None,
            'q_ae_end': self.q_ae_end.tolist() if self.q_ae_end is not None else None
        }
        torch.save(model_info, filepath)

    @staticmethod
    def load_model(filepath, teacher_model, student_model, autoencoder_model):
        """ Load the entire model including sub-models and parameters """
        model_info = torch.load(filepath)
        model = UnifiedAnomalyDetectionModel(
            teacher=teacher_model,
            student=student_model,
            autoencoder=autoencoder_model,
            out_channels=model_info['out_channels'],
            teacher_mean=torch.tensor(model_info['teacher_mean']),
            teacher_std=torch.tensor(model_info['teacher_std']),
            q_st_start=torch.tensor(model_info['q_st_start']) if model_info['q_st_start'] is not None else None,
            q_st_end=torch.tensor(model_info['q_st_end']) if model_info['q_st_end'] is not None else None,
            q_ae_start=torch.tensor(model_info['q_ae_start']) if model_info['q_ae_start'] is not None else None,
            q_ae_end=torch.tensor(model_info['q_ae_end']) if model_info['q_ae_end'] is not None else None
        )
        model.load_state_dict(model_info['model_state_dict'])
        return model
    


"""Model taining example usage - google colab colab

from tutorials.mct_model_garden.models_pytorch.Efficient_Anomaly_Det import get_pdn_small, get_autoencoder
from tutorials.resources.utils.efficient_ad_utils import train_ad

dataset_path = './mvtec_anomaly_detection'
sub_dataset = 'bottle'
train_steps = 70000
out_channels = 384
image_size = 256
teacher_weights = 'drive/MyDrive/anom/teacher_final.pth'
teacher = get_pdn_small(out_channels)
student = get_pdn_small(2 * out_channels)
loaded_model = torch.load(teacher_weights, map_location='cpu')
# Extract the state_dict from the loaded model
state_dict = loaded_model.state_dict()
teacher.load_state_dict(state_dict)
autoencoder = get_autoencoder(out_channels)

train_ad(train_steps, dataset_path, sub_dataset, autoencoder, teacher, student)"""