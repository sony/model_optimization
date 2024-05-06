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
import json
import cv2
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Dict, Tuple, Callable, Any
import random


def coco80_to_coco91(x: np.ndarray) -> np.ndarray:
    """
    Converts COCO 80-class indices to COCO 91-class indices.

    Args:
        x (numpy.ndarray): An array of COCO 80-class indices.

    Returns:
        numpy.ndarray: An array of corresponding COCO 91-class indices.
    """
    coco91Indexs = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
         63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90])
    return coco91Indexs[x.astype(np.int32)]


def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip bounding boxes to stay within the image boundaries.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    boxes[..., 0] = np.clip(boxes[..., 0], a_min=0, a_max=h)
    boxes[..., 1] = np.clip(boxes[..., 1], a_min=0, a_max=w)
    boxes[..., 2] = np.clip(boxes[..., 2], a_min=0, a_max=h)
    boxes[..., 3] = np.clip(boxes[..., 3], a_min=0, a_max=w)
    return boxes


def scale_boxes(boxes: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int,
                preserve_aspect_ratio: bool) -> np.ndarray:
    """
    Scale and offset bounding boxes based on model output size and original image size.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    # Scale and offset boxes
    boxes[..., 0] = (boxes[..., 0] * H - deltaH) * scale_H
    boxes[..., 1] = (boxes[..., 1] * W - deltaW) * scale_W
    boxes[..., 2] = (boxes[..., 2] * H - deltaH) * scale_H
    boxes[..., 3] = (boxes[..., 3] * W - deltaW) * scale_W

    # Clip boxes
    boxes = clip_boxes(boxes, h_image, w_image)

    return boxes


def format_results(outputs: List, img_ids: List, orig_img_dims: List, output_resize: Dict) -> List[Dict]:
    """
    Format model outputs into a list of detection dictionaries.

    Args:
        outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
        img_ids (list): List of image IDs corresponding to each output.
        orig_img_dims (list): List of tuples representing the original image dimensions (h, w) for each output.
        output_resize (Dict): Contains the resize information to map between the model's
                 output and the original image dimensions.

    Returns:
        list: A list of detection dictionaries, each containing information about the detected object.
    """
    detections = []
    h_model, w_model = output_resize['shape']
    preserve_aspect_ratio = output_resize['aspect_ratio_preservation']

    # Process model outputs and convert to detection format
    for idx, output in enumerate(outputs):
        image_id = img_ids[idx]
        scores = output[1].numpy().squeeze()  # Extract scores
        labels = (coco80_to_coco91(
            output[2].numpy())).squeeze()  # Convert COCO 80-class indices to COCO 91-class indices
        boxes = output[0].numpy().squeeze()  # Extract bounding boxes
        boxes = scale_boxes(boxes, orig_img_dims[idx][0], orig_img_dims[idx][1], h_model, w_model,
                            preserve_aspect_ratio)

        for score, label, box in zip(scores, labels, boxes):
            detection = {
                "image_id": image_id,
                "category_id": label,
                "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
                "score": score
            }
            detections.append(detection)

    return detections


# COCO evaluation class
class CocoEval:
    def __init__(self, path2json: str, output_resize: Dict = None):
        """
        Initialize the CocoEval class.

        Args:
            path2json (str): Path to the COCO JSON file containing ground truth annotations.
            output_resize (Dict): Contains the resize information to map between the model's output and the original
             image dimensions. The dict consists of:
                  {"shape": (height, weight),
                   "aspect_ratio_preservation": bool}
        """
        # Load ground truth annotations
        self.coco_gt = COCO(path2json)

        # A list of reformatted model outputs
        self.all_detections = []

        # Resizing information to map between the model's output and the original image dimensions
        self.output_resize = output_resize if output_resize else {'shape': (1, 1), 'aspect_ratio_preservation': False}

    def add_batch_detections(self, outputs: Tuple[List, List, List, List], targets: List[Dict]):
        """
        Add batch detections to the evaluation.

        Args:
            outputs (list): List of model outputs, typically containing bounding boxes, scores, and labels.
            targets (list): List of ground truth annotations for the batch.
        """
        img_ids, _outs = [], []
        orig_img_dims = []
        for idx, t in enumerate(targets):
            if len(t) > 0:
                img_ids.append(t[0]['image_id'])
                orig_img_dims.append(t[0]['orig_img_dims'])
                _outs.append([outputs[0][idx], outputs[1][idx], outputs[2][idx], outputs[3][idx]])

        batch_detections = format_results(_outs, img_ids, orig_img_dims, self.output_resize)

        self.all_detections.extend(batch_detections)

    def result(self) -> List[float]:
        """
        Calculate and print evaluation results.

        Returns:
            list: COCO evaluation statistics.
        """
        # Initialize COCO evaluation object
        self.coco_dt = self.coco_gt.loadRes(self.all_detections)
        coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')

        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Print mAP results
        print("mAP: {:.4f}".format(coco_eval.stats[0]))

        return coco_eval.stats

    def reset(self):
        """
        Reset the list of detections to prepare for a new evaluation.
        """
        self.all_detections = []


def load_and_preprocess_image(image_path: str, preprocess: Callable) -> np.ndarray:
    """
    Load and preprocess an image from a given file path.

    Args:
        image_path (str): Path to the image file.
        preprocess (function): Preprocessing function to apply to the loaded image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    image = preprocess(image)
    return image


class CocoDataset:
    def __init__(self, dataset_folder: str, annotation_file: str, preprocess: Callable):
        """
        A dataset class for handling COCO dataset images and annotations.

        Args:
            dataset_folder (str): The path to the folder containing COCO dataset images.
            annotation_file (str): The path to the COCO annotation file in JSON format.
            preprocess (Callable): A function for preprocessing images.
        """
        self.dataset_folder = dataset_folder
        self.preprocess = preprocess

        # Load COCO annotations from a JSON file (e.g., 'annotations.json')
        with open(annotation_file, 'r') as f:
            self.coco_annotations = json.load(f)

        # Initialize a dictionary to store annotations grouped by image ID
        self.annotations_by_image = {}

        # Iterate through the annotations and group them by image ID
        for annotation in self.coco_annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(annotation)

        # Initialize a list to collect images and annotations for the current batch
        self.total_images = len(self.coco_annotations['images'])

    def __len__(self):
        return self.total_images

    def __getitem__(self, item_index):
        """
        Returns the preprocessed image and its corresponding annotations.

        Args:
            item_index: Index of the item to retrieve.

        Returns:
            Tuple containing the preprocessed image and its annotations.
        """
        image_info = self.coco_annotations['images'][item_index]
        image_id = image_info['id']
        image = load_and_preprocess_image(os.path.join(self.dataset_folder, image_info['file_name']), self.preprocess)
        annotations = self.annotations_by_image.get(image_id, [])
        if len(annotations) > 0:
            annotations[0]['orig_img_dims'] = (image_info['height'], image_info['width'])
        return image, annotations

    def sample(self, batch_size):
        """
        Samples a batch of images and their corresponding annotations from the dataset.

        Returns:
            Tuple containing a batch of preprocessed images and their annotations.
        """
        batch_images = []
        batch_annotations = []

        # Sample random image indexes
        random_idx = random.sample(range(self.total_images), batch_size)

        # Get the corresponding items from dataset
        for idx in random_idx:
            batch_images.append(self[idx][0])
            batch_annotations.append(self[idx][1])

        return np.array(batch_images), batch_annotations


class DataLoader:
    def __init__(self, dataset: List[Tuple], batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = 0
        self.inds = list(range(len(dataset)))

    def __iter__(self):
        self.count = 0
        if self.shuffle:
            random.shuffle(self.inds)

        return self

    def __next__(self):
        if self.count >= len(self.dataset):
            raise StopIteration

        batch_images = []
        batch_annotations = []

        while len(batch_images) < self.batch_size and self.count < len(self.dataset):
            index = self.inds[self.count]
            image, annotations = self.dataset[index]
            batch_images.append(image)
            batch_annotations.append(annotations)
            self.count += 1

        return np.array(batch_images), batch_annotations


def coco_dataset_generator(dataset_folder: str, annotation_file: str, preprocess: Callable,
                           batch_size: int = 1) -> Tuple:

    """
    Generator function for loading and preprocessing images and their annotations from a COCO-style dataset.

    Args:
        dataset_folder (str): Path to the dataset folder containing image files.
        annotation_file (str): Path to the COCO-style annotation JSON file.
        preprocess (function): Preprocessing function to apply to each loaded image.
        batch_size (int): The desired batch size.

    Yields:
        Tuple[numpy.ndarray, list]: A tuple containing a batch of images (as a NumPy array) and a list of annotations
        for each image in the batch.
    """
    # Load COCO annotations from a JSON file (e.g., 'annotations.json')
    with open(annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    # Initialize a dictionary to store annotations grouped by image ID
    annotations_by_image = {}

    # Iterate through the annotations and group them by image ID
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Initialize a list to collect images and annotations for the current batch
    batch_images = []
    batch_annotations = []
    total_images = len(coco_annotations['images'])

    # Iterate through the images and create a list of tuples (image, annotations)
    for image_count, image_info in enumerate(coco_annotations['images']):
        image_id = image_info['id']
        # Load and preprocess the image (you can use your own image loading logic)
        image = load_and_preprocess_image(os.path.join(dataset_folder, image_info['file_name']), preprocess)
        annotations = annotations_by_image.get(image_id, [])
        if len(annotations) > 0:
            annotations[0]['orig_img_dims'] = (image_info['height'], image_info['width'])

            # Add the image and annotations to the current batch
            batch_images.append(image)
            batch_annotations.append(annotations)

            # Check if the current batch is of the desired batch size
            if len(batch_images) == batch_size:
                # Yield the current batch
                yield np.array(batch_images), batch_annotations

                # Reset the batch lists for the next batch
                batch_images = []
                batch_annotations = []

        # After processing all images, yield any remaining images in the last batch
        if len(batch_images) > 0 and (total_images == image_count + 1):
            yield np.array(batch_images), batch_annotations


def model_predict(model: Any,
                  inputs: np.ndarray) -> Tuple[List, List, List, List]:
    """
    Perform inference using the provided model on the given inputs.

    This function serves as the default method for inference if no specific model inference function is provided.

    Args:
        model (Any): The model used for inference.
        inputs (np.ndarray): Input data to perform inference on.

    Returns:
        Tuple[List, List, List, List]: Tuple containing lists of predictions.
    """
    return model(inputs)


def coco_evaluate(model: Any, preprocess: Callable, dataset_folder: str, annotation_file: str, batch_size: int,
                  output_resize: tuple, model_inference: Callable = model_predict) -> dict:
    """
    Evaluate a model on the COCO dataset.

    Args:
    - model (Any): The model to evaluate.
    - preprocess (Callable): Preprocessing function to be applied to images.
    - dataset_folder (str): Path to the folder containing COCO dataset images.
    - annotation_file (str): Path to the COCO annotation file.
    - batch_size (int): Batch size for evaluation.
    - output_resize (tuple): Tuple representing the output size after resizing.
    - model_inference (Callable): Model inference function. model_predict will be used by default.

    Returns:
    - dict: Evaluation results.

    """
    # Load COCO evaluation set
    coco_dataset = CocoDataset(dataset_folder=dataset_folder,
                               annotation_file=annotation_file,
                               preprocess=preprocess)
    coco_loader = DataLoader(coco_dataset, batch_size)

    # Initialize the evaluation metric object
    coco_metric = CocoEval(annotation_file, output_resize)

    # Iterate and the evaluation set
    for batch_idx, (images, targets) in enumerate(coco_loader):

        # Run inference on the batch
        outputs = model_inference(model, images)

        # Add the model outputs to metric object (a dictionary of outputs after postprocess: boxes, scores & classes)
        coco_metric.add_batch_detections(outputs, targets)
        if (batch_idx + 1) % 100 == 0:
            print(f'processed {(batch_idx + 1) * batch_size} images')

    return coco_metric.result()
