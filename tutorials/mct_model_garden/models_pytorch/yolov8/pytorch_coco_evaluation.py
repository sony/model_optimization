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
import json
import cv2
import os
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Dict, Tuple, Callable, Any

from torch import nn

from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation import coco_dataset_generator, CocoEval


def coco_evaluate(model: nn.Module,
                  preprocess: Callable,
                  dataset_folder: str,
                  annotation_file: str,
                  batch_size: int,
                  output_resize: tuple,
                  device: str,
                  data_dtype: torch.dtype) -> dict:
    """
    Evaluate a model on the COCO dataset.

    Args:
    - model (nn.Module): Model to validate.
    - preprocess (Callable): Preprocessing function to be applied to images.
    - dataset_folder (str): Path to the folder containing COCO dataset images.
    - annotation_file (str): Path to the COCO annotation file.
    - batch_size (int): Batch size for evaluation.
    - output_resize (tuple): Tuple representing the output size after resizing.
    - device (str): The current device set for PyTorch operations.
    - data_dtype (torch.dtype): The dtype set for model inputs.

    Returns:
    - dict: Evaluation results.

    """
    # Load COCO evaluation set
    val_dataset = coco_dataset_generator(dataset_folder=dataset_folder,
                                         annotation_file=annotation_file,
                                         preprocess=preprocess,
                                         batch_size=batch_size)

    # Initialize the evaluation metric object
    coco_metric = CocoEval(annotation_file, output_resize)

    # Iterate and the evaluation set
    for batch_idx, (images, targets) in enumerate(val_dataset):

        # Run Pytorch inference on the batch
        images = torch.from_numpy(images).to(device=device, dtype=data_dtype)
        outputs = model(images)

        # Detach outputs and move to cpu
        output_list = [output.detach().cpu() for output in outputs]

        # Add the model outputs to metric object (a dictionary of outputs after postprocess: boxes, scores & classes)
        coco_metric.add_batch_detections(output_list, targets)
        if (batch_idx + 1) % 100 == 0:
            print(f'processed {(batch_idx + 1) * batch_size} images')

    return coco_metric.result()
