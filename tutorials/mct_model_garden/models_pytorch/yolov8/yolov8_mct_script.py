import copy
from typing import Iterator, Tuple, List

import torch

import model_compression_toolkit as mct
from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import coco_dataset_generator
from tutorials.mct_model_garden.models_keras.yolov8.yolov8_preprocess import yolov8_preprocess_chw_transpose
from tutorials.mct_model_garden.models_pytorch.yolov8.pytorch_coco_evaluation import coco_evaluate
from tutorials.mct_model_garden.models_pytorch.yolov8.yolov8 import yolov8_pytorch, PostProcessWrapper, \
    DetectionModelPyTorch, yaml_load

MP_WEIGHT_SHRINK = 0.75
BATCH_SIZE = 4
INPUT_RESOLUTION = 640
REPRESENTATIVE_DATASET_FOLDER = '/data/projects/swat/datasets_src/COCO/images/val2017'
REPRESENTATIVE_DATASET_ANNOTATION_FILE = '/data/projects/swat/datasets_src/COCO/annotations/instances_val2017.json'
output_resize = {'shape': (INPUT_RESOLUTION, INPUT_RESOLUTION), 'aspect_ratio_preservation': True}

n_iters = 20
score_threshold = 0.001
iou_threshold = 0.7
max_detections = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float

# save model
# model, cfg_dict = yolov8_pytorch(model_yaml="yolov8n.yaml")
# org_dict = torch.load("/Vols/vol_design/tools/swat/users/ariell/repos/my_fork/yolov8_tut/yolov8n.pt")
# model.load_state_dict(org_dict['model'].state_dict(), strict=False)
# model.save_pretrained("pytorch_yolov8n_640x640", cfg=cfg_dict)

# load
cfg_dict = yaml_load("yolov8n.yaml", append_filename=True)  # model dict
model = DetectionModelPyTorch.from_pretrained("pytorch_yolov8n_640x640", cfg=cfg_dict)

model = model.eval()
model_pp = PostProcessWrapper(model=model,
                              score_threshold=score_threshold,
                              iou_threshold=iou_threshold,
                              max_detections=max_detections).to(device=device)

# float eval
eval_results = coco_evaluate(model=model_pp,
                             dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                             annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                             preprocess=yolov8_preprocess_chw_transpose,
                             output_resize=output_resize,
                             batch_size=BATCH_SIZE,
                             device=device,
                             data_dtype=dtype)

# Load representative dataset
representative_dataset = coco_dataset_generator(dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                                                annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                                                preprocess=yolov8_preprocess_chw_transpose,
                                                batch_size=BATCH_SIZE)


# Define representative dataset generator
def get_representative_dataset(n_iter: int, dataset_loader: Iterator[Tuple]):
    def representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset_loader)
        for _ in range(n_iter):
            yield [next(ds_iter)[0]]

    return representative_dataset


# Get representative dataset generator
representative_dataset_gen = get_representative_dataset(n_iters, representative_dataset)

tpc = mct.get_target_platform_capabilities("pytorch", 'imx500', target_platform_version='v1')
ptq_config = mct.core.CoreConfig(quantization_config=mct.core.QuantizationConfig(
    shift_negative_activation_correction=True))

ptq_model = copy.deepcopy(model)

ptq_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=ptq_model,
                                                          representative_data_gen=representative_dataset_gen,
                                                          core_config=ptq_config,
                                                          target_platform_capabilities=tpc)

# add PostProcess
ptq_model_pp = PostProcessWrapper(model=ptq_model,
                                  score_threshold=score_threshold,
                                  iou_threshold=iou_threshold,
                                  max_detections=max_detections).to(device=device)

# ptq eval
ptq_eval_results = coco_evaluate(model=ptq_model_pp,
                                 dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                                 annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                                 preprocess=yolov8_preprocess_chw_transpose,
                                 output_resize=output_resize,
                                 batch_size=BATCH_SIZE,
                                 device=device,
                                 data_dtype=dtype)

mp_model = copy.deepcopy(model)
mp_quant_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=5, use_hessian_based_scores=False)
mp_config = mct.core.CoreConfig(mixed_precision_config=mp_quant_config,
                                quantization_config=mct.core.QuantizationConfig(
                                    shift_negative_activation_correction=True))
resource_utilization_data = mct.core.pytorch_resource_utilization_data(in_model=mp_model,
                                                                       representative_data_gen=
                                                                       representative_dataset_gen,
                                                                       core_config=mp_config,
                                                                       target_platform_capabilities=tpc)
resource_utilization = mct.core.ResourceUtilization(resource_utilization_data.weights_memory * MP_WEIGHT_SHRINK)

mp_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=mp_model,
                                                         representative_data_gen=representative_dataset_gen,
                                                         target_resource_utilization=resource_utilization,
                                                         core_config=mp_config,
                                                         target_platform_capabilities=tpc)
# add PostProcess
mp_model_pp = PostProcessWrapper(model=mp_model,
                                 score_threshold=score_threshold,
                                 iou_threshold=iou_threshold,
                                 max_detections=max_detections).to(device=device)

# mp eval
mp_eval_results = coco_evaluate(model=mp_model_pp,
                                dataset_folder=REPRESENTATIVE_DATASET_FOLDER,
                                annotation_file=REPRESENTATIVE_DATASET_ANNOTATION_FILE,
                                preprocess=yolov8_preprocess_chw_transpose,
                                output_resize=output_resize,
                                batch_size=BATCH_SIZE,
                                device=device,
                                data_dtype=dtype)
