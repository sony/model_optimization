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
from pycocotools import mask as mask_utils
from tqdm import tqdm

from tutorials.mct_model_garden.evaluation_metrics.coco_evaluation_utils import scale_boxes, scale_coords
from ..models_pytorch.yolov8.yolov8_preprocess import yolov8_preprocess_chw_transpose
from ..models_pytorch.yolov8.postprocess_yolov8_seg import process_masks, postprocess_yolov8_inst_seg


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


# COCO evaluation class
class CocoEval:
    def __init__(self, path2json: str, output_resize: Dict = None, task: str = 'Detection'):
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

        # Set the task type (Detection/Segmentation/Keypoints)
        self.task = task

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
                _outs.append([o[idx] for o in outputs])

        batch_detections = self.format_results(_outs, img_ids, orig_img_dims, self.output_resize)

        self.all_detections.extend(batch_detections)

    def result(self) -> List[float]:
        """
        Calculate and print evaluation results.

        Returns:
            list: COCO evaluation statistics.
        """
        # Initialize COCO evaluation object
        self.coco_dt = self.coco_gt.loadRes(self.all_detections)
        if self.task == 'Detection':
            coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'bbox')
        elif self.task == 'Keypoints':
            coco_eval = COCOeval(self.coco_gt, self.coco_dt, 'keypoints')
        else:
            raise Exception("Unsupported task type of CocoEval")

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

    def format_results(self, outputs: List, img_ids: List, orig_img_dims: List, output_resize: Dict) -> List[Dict]:
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
        normalized_coords = output_resize.get('normalized_coords', True)
        align_center = output_resize.get('align_center', True)

        if self.task == 'Detection':
            # Process model outputs and convert to detection format
            for idx, output in enumerate(outputs):
                image_id = img_ids[idx]
                scores = output[1].numpy().squeeze()  # Extract scores
                labels = (coco80_to_coco91(
                    output[2].numpy())).squeeze()  # Convert COCO 80-class indices to COCO 91-class indices
                boxes = output[0].numpy().squeeze()  # Extract bounding boxes
                boxes = scale_boxes(boxes, orig_img_dims[idx][0], orig_img_dims[idx][1], h_model, w_model,
                                    preserve_aspect_ratio, align_center, normalized_coords)

                for score, label, box in zip(scores, labels, boxes):
                    detection = {
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [box[1], box[0], box[3] - box[1], box[2] - box[0]],
                        "score": score
                    }
                    detections.append(detection)

        elif self.task == 'Keypoints':
            for output, image_id, (w_orig, h_orig) in zip(outputs, img_ids, orig_img_dims):

                bbox, scores, kpts = output

                # Add detection results to predicted_keypoints list
                if kpts.shape[0]:
                    kpts = kpts.reshape(-1, 17, 3)
                    kpts = scale_coords(kpts, h_orig, w_orig, 640, 640, True)
                    for ind, k in enumerate(kpts):
                        detections.append({
                            'category_id': 1,
                            'image_id': image_id,
                            'keypoints': k.reshape(51).tolist(),
                            'score': scores.tolist()[ind] if isinstance(scores.tolist(), list) else scores.tolist()
                        })

        return detections

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
                  output_resize: tuple, model_inference: Callable = model_predict, task: str = 'Detection') -> dict:
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
    coco_metric = CocoEval(annotation_file, output_resize, task)

    # Iterate and the evaluation set
    for batch_idx, (images, targets) in enumerate(coco_loader):

        # Run inference on the batch
        outputs = model_inference(model, images)

        # Add the model outputs to metric object (a dictionary of outputs after postprocess: boxes, scores & classes)
        coco_metric.add_batch_detections(outputs, targets)
        if (batch_idx + 1) % 100 == 0:
            print(f'processed {(batch_idx + 1) * batch_size} images')

    return coco_metric.result()

def masks_to_coco_rle(masks, boxes, image_id, height, width, scores, classes, mask_threshold):
    """
    Converts masks to COCO RLE format and compiles results including bounding boxes and scores.

    Args:
        masks (list of np.ndarray): List of segmentation masks.
        boxes (list of np.ndarray): List of bounding boxes corresponding to the masks.
        image_id (int): Identifier for the image being processed.
        height (int): Height of the image.
        width (int): Width of the image.
        scores (list of float): Confidence scores for each detection.
        classes (list of int): Class IDs for each detection.

    Returns:
        list of dict: Each dictionary contains the image ID, category ID, bounding box,
                      score, and segmentation in RLE format.
    """
    results = []
    for i, (mask, box) in enumerate(zip(masks, boxes)):

        binary_mask = np.asfortranarray((mask > mask_threshold).astype(np.uint8))
        rle = mask_utils.encode(binary_mask)
        rle['counts'] = rle['counts'].decode('ascii')

        x_min, y_min, x_max, y_max = box[1], box[0], box[3], box[2]
        box_width = x_max - x_min
        box_height = y_max - y_min

        adjusted_category_id = coco80_to_coco91(np.array([classes[i]]))[0]
        
        result = {
            "image_id": int(image_id),  # Convert to int if not already
            "category_id": int(adjusted_category_id),  # Ensure type is int
            "bbox": [float(x_min), float(y_min), float(box_width), float(box_height)],
            "score": float(scores[i]),  # Ensure type is float
            "segmentation": rle
        }
        results.append(result)
    return results

def save_results_to_json(results, file_path):
    """
    Saves the results data to a JSON file.

    Args:
        results (list of dict): The results data to be saved.
        file_path (str): The path to the file where the results will be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(results, f)

def evaluate_seg_model(annotation_file, results_file):
    """
    Evaluate the model's segmentation performance using the COCO evaluation metrics.

    This function loads the ground truth annotations and the detection results from specified files,
    filters the annotations to include only those images present in the detection results, and then
    performs the COCO evaluation.

    Args:
        annotation_file (str): The file path for the COCO format ground truth annotations.
        results_file (str): The file path for the detection results in COCO format.

    The function prints out the evaluation summary which includes average precision and recall
    across various IoU thresholds and object categories.
    """
   
    coco_gt = COCO(annotation_file)
    coco_dt = coco_gt.loadRes(results_file)
    
    # Extract image IDs from the results file
    with open(results_file, 'r') as file:
        results_data = json.load(file)
    result_img_ids = {result['image_id'] for result in results_data}
    
    # Filter annotations to include only those images present in the results file
    coco_gt.imgs = {img_id: coco_gt.imgs[img_id] for img_id in result_img_ids if img_id in coco_gt.imgs}
    coco_gt.anns = {ann_id: coco_gt.anns[ann_id] for ann_id in list(coco_gt.anns.keys()) if coco_gt.anns[ann_id]['image_id'] in result_img_ids}
    
    # Evaluate only for the filtered images
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.imgIds = list(result_img_ids)  # Ensure evaluation is only on the filtered image IDs
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def evaluate_yolov8_segmentation(model, model_predict_func, data_dir, data_type='val2017', img_ids_limit=800, output_file='results.json',iou_thresh=0.7, conf=0.001, max_dets=300,mask_thresh=0.55):
    """
    Evaluate YOLOv8 model for instance segmentation on COCO dataset.

    Parameters:
    - model: The YOLOv8 model to be evaluated.
    - model_predict_func: A function to execute the model preidction
    - data_dir: The directory containing the COCO dataset.
    - data_type: The type of dataset to evaluate against (default is 'val2017').
    - img_ids_limit: The maximum number of images to evaluate (default is 800).
    - output_file: The name of the file to save the results (default is 'results.json').

    Returns:
    - None
    """
    model_input_size = (640, 640)
    model.eval()

    ann_file = os.path.join(data_dir, 'annotations', f'instances_{data_type}.json')
    coco = COCO(ann_file)

    img_ids = coco.getImgIds()
    img_ids = img_ids[:img_ids_limit]  # Adjust number of images to evaluate against
    results = []
    for img_id in tqdm(img_ids, desc="Processing Images"):
        img = coco.loadImgs(img_id)[0]
        image_path = os.path.join(data_dir, data_type, img["file_name"])

        # Preprocess the image
        input_img = load_and_preprocess_image(image_path, yolov8_preprocess_chw_transpose).astype('float32')

        # Run the model
        output = model_predict_func(model, input_img)

        #run post processing (nms)
        boxes, scores, classes, masks = postprocess_yolov8_inst_seg(outputs=output, conf=conf, iou_thres=iou_thresh, max_out_dets=max_dets)

        if boxes.size == 0:  
            continue

        orig_img = load_and_preprocess_image(image_path, lambda x: x)
        boxes = scale_boxes(boxes, orig_img.shape[0], orig_img.shape[1], 640, 640, True, False)
        pp_masks = process_masks(masks, boxes, orig_img.shape, model_input_size)

        #convert output to coco readable
        image_results = masks_to_coco_rle(pp_masks, boxes, img_id, orig_img.shape[0], orig_img.shape[1], scores, classes, mask_thresh)
        results.extend(image_results)

    save_results_to_json(results, output_file)
    evaluate_seg_model(ann_file, output_file)
