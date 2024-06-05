from enum import Enum
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt

def nms(dets: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5, max_out_dets: int = 300) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on detected bounding boxes.

    Args:
        dets (np.ndarray): Array of bounding box coordinates of shape (N, 4) representing [y1, x1, y2, x2].
        scores (np.ndarray): Array of confidence scores associated with each bounding box.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.5.
        max_out_dets (int, optional): Maximum number of output detections to keep. Default is 300.

    Returns:
        List[int]: List of indices representing the indices of the bounding boxes to keep after NMS.

    """
    y1, x1 = dets[:, 0], dets[:, 1]
    y2, x2 = dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep[:max_out_dets]

def combined_nms(batch_boxes, batch_scores, iou_thres: float = 0.5, conf: float = 0.001, max_out_dets: int = 300):

    nms_results = []
    for boxes, scores in zip(batch_boxes, batch_scores):

        xc = np.argmax(scores, 1)
        xs = np.amax(scores, 1)
        x = np.concatenate([boxes, np.expand_dims(xs, 1), np.expand_dims(xc, 1)], 1)

        xi = xs > conf
        x = x[xi]

        x = x[np.argsort(-x[:, 4])[:8400]]
        scores = x[:, 4]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        offset = x[:, 5] * 640
        boxes = x[..., :4] + np.expand_dims(offset, 1)

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_classes = x[:, 5]
        nms_bbox = x[:, :4]
        nms_scores = x[:, 4]

        nms_results.append((nms_bbox, nms_scores, nms_classes))

    return nms_results


class BoxFormat(Enum):
    YMIM_XMIN_YMAX_XMAX = 'ymin_xmin_ymax_xmax'
    XMIM_YMIN_XMAX_YMAX = 'xmin_ymin_xmax_ymax'
    XMIN_YMIN_W_H = 'xmin_ymin_width_height'
    XC_YC_W_H = 'xc_yc_width_height'


def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
    """
    changes the box from one format to another (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX )
    also support in same format mode (returns the same format)

    :param boxes:
    :param orig_format:
    :return: box in format YMIM_XMIN_YMAX_XMAX
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        boxes[:, 2] += boxes[:, 0]  # convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # convert height to ymax
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")

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


def scale_boxes(boxes: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool, normalized: bool = True) -> np.ndarray:
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

    nh, nw = (H, W) if normalized else (1, 1)

    # Scale and offset boxes
    boxes[..., 0] = (boxes[..., 0] * nh - deltaH) * scale_H
    boxes[..., 1] = (boxes[..., 1] * nw - deltaW) * scale_W
    boxes[..., 2] = (boxes[..., 2] * nh - deltaH) * scale_H
    boxes[..., 3] = (boxes[..., 3] * nw - deltaW) * scale_W

    # Clip boxes
    boxes = clip_boxes(boxes, h_image, w_image)

    return boxes


def scale_coords(kpts: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool) -> np.ndarray:
    """
    Scale and offset keypoints based on model output size and original image size.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
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
    kpts[..., 0] = (kpts[..., 0]  - deltaH) * scale_H
    kpts[..., 1] = (kpts[..., 1] - deltaW) * scale_W

    # Clip boxes
    kpts = clip_coords(kpts, h_image, w_image)

    return kpts

def clip_coords(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip keypoints to stay within the image boundaries.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    kpts[..., 0] = np.clip(kpts[..., 0], a_min=0, a_max=h)
    kpts[..., 1] = np.clip(kpts[..., 1], a_min=0, a_max=w)
    return kpts

def postprocess_yolov8_detection(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       conf: float = 0.001,
                       iou_thres: float = 0.7,
                       max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess the outputs of a YOLOv8 model for object detection and pose estimation.

    Args:
        outputs (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing the model outputs for bounding boxes,
            class predictions, and keypoint predictions.
        conf (float, optional): Confidence threshold for bounding box predictions. Default is 0.001.
        iou_thres (float, optional): IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).
            Default is 0.7.
        max_out_dets (int, optional): Maximum number of output detections to keep after NMS. Default is 300.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the post-processed bounding boxes,
            their corresponding scores, and keypoints.

    """

    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)

    xd = detect_out.transpose([0, 2, 1])

    return combined_nms(xd[..., :4], xd[..., 4:84], iou_thres, conf, max_out_dets)


def postprocess_yolov8_keypoints(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       conf: float = 0.001,
                       iou_thres: float = 0.7,
                       max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess the outputs of a YOLOv8 model for object detection and pose estimation.

    Args:
        outputs (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing the model outputs for bounding boxes,
            class predictions, and keypoint predictions.
        conf (float, optional): Confidence threshold for bounding box predictions. Default is 0.001.
        iou_thres (float, optional): IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).
            Default is 0.7.
        max_out_dets (int, optional): Maximum number of output detections to keep after NMS. Default is 300.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the post-processed bounding boxes,
            their corresponding scores, and keypoints.

    """
    kpt_shape = (17, 3)
    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, kpts = outputs
    dbox = dist2bbox_yolo_v8(y_bb, np.expand_dims(a, 0), xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)
    # additional part for pose estimation
    ndim = kpt_shape[1]
    pred_kpt = kpts.copy()
    if ndim == 3:
        pred_kpt[:, 2::3] = 1 / (1 + np.exp(-pred_kpt[:, 2::3]))  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
    pred_kpt[:, 0::ndim] = (pred_kpt[:, 0::ndim] * 2.0 + (a[0] - 0.5)) * s
    pred_kpt[:, 1::ndim] = (pred_kpt[:, 1::ndim] * 2.0 + (a[1] - 0.5)) * s

    x_batch = np.concatenate([detect_out.transpose([0, 2, 1]), pred_kpt.transpose([0, 2, 1])], 2)
    nms_bbox, nms_scores, nms_kpts = [], [], []
    for x in x_batch:
        x = x[(x[:, 4] > conf)]
        x = x[np.argsort(-x[:, 4])[:8400]]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        boxes = x[..., :4]
        scores = x[..., 4]

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_bbox.append(x[:, :4])
        nms_scores.append(x[:, 4])
        nms_kpts.append(x[:, 5:])

    return nms_bbox, nms_scores, nms_kpts


def postprocess_yolov8_inst_seg(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
                       conf: float = 0.001,
                       iou_thres: float = 0.7,
                       max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, y_masks = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)

    xd = detect_out.transpose([0, 2, 1])

    return combined_nms(xd[..., :4], xd[..., 4:84], iou_thres, conf, max_out_dets)


def make_anchors_yolo_v8(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = np.arange(stop=w) + grid_cell_offset  # shift x
        sy = np.arange(stop=h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox_yolo_v8(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance,2,axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox


def scale_coords(kpts: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool) -> np.ndarray:
    """
    Scale and offset keypoints based on model output size and original image size.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
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
    kpts[..., 0] = (kpts[..., 0]  - deltaH) * scale_H
    kpts[..., 1] = (kpts[..., 1] - deltaW) * scale_W

    # Clip boxes
    kpts = clip_coords(kpts, h_image, w_image)

    return kpts


def clip_coords(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip keypoints to stay within the image boundaries.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    kpts[..., 0] = np.clip(kpts[..., 0], a_min=0, a_max=h)
    kpts[..., 1] = np.clip(kpts[..., 1], a_min=0, a_max=w)
    return kpts


class COCODrawer:
    def __init__(self, categories_file):
        self.categories = self.get_categories(categories_file)

    def get_categories(self, filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def draw_bounding_box(self, img, annotation, class_id, ax):
        x_min, y_min = int(annotation[1]), int(annotation[0])
        x_max, y_max = int(annotation[3]), int(annotation[2])
        text = self.categories[int(class_id)]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        ax.text(x_min, y_min, text, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    def draw_keypoints(self, img, keypoints):
        skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 7], [7, 9], [6, 8],  # Arms
            [8, 10], [5, 11], [6, 12], [11, 12],  # Body
            [11, 13], [12, 14], [13, 15], [14, 16]  # Legs
        ]

        # Draw skeleton lines
        for connection in skeleton:
            start_point = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
            end_point = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
            cv2.line(img, start_point, end_point, (255, 0, 0), 2)

        # Draw keypoints as colored circles
        for point in keypoints:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    def annotate_image(self, img, b, s, c, k, scale, ax):
        for index, row in enumerate(b):
            if s[index] > 0.55:
                self.draw_bounding_box(img, row * scale, c[index], ax)
                if k is not None:
                    self.draw_keypoints(img, k[index] * scale)

    def plot_image(self, img, b, s, c, k=None):
        fig, ax = plt.subplots()
        self.annotate_image(img, b, s, c, k, scale=1, ax=ax)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title(img.shape)
        return fig, ax
