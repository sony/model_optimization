
import os
from tqdm import tqdm
import numpy as np
import torch

from torchvision import transforms
from sklearn.metrics import roc_auc_score
import tifffile
from tutorials.resources.utils.efficient_ad_utils import ImageFolderWithPath, predict_combined

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


def benchmark(unified_model, name, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    """Benchmark the model by testing it on a dataset and printing the AUC score."""
    dataset_path = './mvtec_anomaly_detection'
    test_output_dir = os.path.join('output', 'anomaly_maps', name, 'bottle', 'test')
    test_set = ImageFolderWithPath(os.path.join(dataset_path, 'bottle', 'test'))
    unified_model.eval()
    auc = test(test_set=test_set, unified_model=unified_model, test_output_dir=test_output_dir, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))

def test(test_set, unified_model, test_output_dir=None, desc='Running inference'):
    """Test the model and calculate the AUC score."""
    y_true, y_score = [], []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width, orig_height = image.size
        image = DEFAULT_TRANSFORM(image)[None]  # Add batch dimension
        if torch.cuda.is_available():
            image = image.cuda()
        map_combined = predict_combined(image, unified_model, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None)
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].detach().cpu().numpy()
        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir:
            img_nm = os.path.split(path)[1].split('.')[0]
            defect_dir = os.path.join(test_output_dir, defect_class)
            os.makedirs(defect_dir, exist_ok=True)
            tifffile.imwrite(os.path.join(defect_dir, img_nm + '.tiff'), map_combined)
        y_true.append(0 if defect_class == 'good' else 1)
        y_score.append(np.max(map_combined))
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100