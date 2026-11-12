"""YOLOv8n COCO detector for a selected set of object classes.

The MPK returns three DFL box heads followed by three 80-class score heads.
This module keeps the model setup, decoding, drawing, and optional zone alert
policy separate from the solar-panel segmentation code.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("solar_panel_pipeline.animals")
INPUT_SIZE = 640
GRIDS = (80, 40, 20)
STRIDES = (8.0, 16.0, 32.0)
DFL_BINS = np.arange(16, dtype=np.float32)
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


def _anchors():
    points, strides = [], []
    for grid, stride in zip(GRIDS, STRIDES):
        axis = np.arange(grid, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(axis, axis)
        points.append(np.stack((xx.ravel(), yy.ravel()), axis=0))
        strides.append(np.full(grid * grid, stride, dtype=np.float32))
    return np.concatenate(points, axis=1), np.concatenate(strides)


ANCHORS, ANCHOR_STRIDES = _anchors()


def letterbox_geometry(width: int, height: int) -> tuple[float, float, float]:
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    return scale, (INPUT_SIZE - width * scale) / 2.0, (INPUT_SIZE - height * scale) / 2.0


def _nchw(tensor: Any, channels: int, grid: int, label: str) -> np.ndarray:
    array = np.asarray(tensor.to_numpy(copy=False), dtype=np.float32)
    if array.shape == (1, grid, grid, channels):
        return array[0].transpose(2, 0, 1)
    if array.shape == (1, channels, grid, grid):
        return array[0]
    raise RuntimeError(f"Unexpected YOLOv8n {label} output shape {array.shape}")


def _nms(boxes, scores, iou_threshold: float, max_detections: int) -> list[int]:
    import cv2

    picked: list[int] = []
    for class_id in sorted(set(int(v) for v in boxes[:, 4])):
        indices = np.flatnonzero(boxes[:, 4].astype(np.int32) == class_id)
        rects = [[float(boxes[i, 0]), float(boxes[i, 1]),
                  float(boxes[i, 2] - boxes[i, 0]), float(boxes[i, 3] - boxes[i, 1])] for i in indices]
        keep = cv2.dnn.NMSBoxes(rects, [float(scores[i]) for i in indices], 0.0,
                                float(iou_threshold), top_k=100)
        if len(keep):
            picked.extend(int(indices[int(i)]) for i in np.asarray(keep).reshape(-1))
    picked.sort(key=lambda i: float(scores[i]), reverse=True)
    return picked[:max(1, int(max_detections))]


def decode_yolov8n(outputs: list[Any], width: int, height: int,
                   class_ids: frozenset[int], threshold: float = 0.35,
                   iou_threshold: float = 0.45, max_detections: int = 30) -> list[dict[str, Any]]:
    """Decode this archive's six raw NHWC heads into original-frame boxes."""
    if len(outputs) != 6:
        shapes = [tuple(getattr(t, "shape", ())) for t in outputs]
        raise RuntimeError(f"YOLOv8n archive should return 6 tensors, got {len(outputs)}: {shapes}")

    box_heads, score_heads = [], []
    for i, grid in enumerate(GRIDS):
        box_heads.append(_nchw(outputs[i], 64, grid, f"box[{i}]").reshape(64, -1))
        score_heads.append(_nchw(outputs[i + 3], 80, grid, f"class[{i}]").reshape(80, -1))
    scores = np.concatenate(score_heads, axis=1)
    # This package names the outputs class_prob. Keep probabilities as-is;
    # accommodate logit-valued archives too, detected from values outside [0,1].
    if scores.min(initial=0.0) < -1e-5 or scores.max(initial=0.0) > 1.00001:
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -80, 80)))
    selected_ids = np.asarray(sorted(class_ids), dtype=np.int32)
    if selected_ids.size == 0 or np.any((selected_ids < 0) | (selected_ids >= 80)):
        raise ValueError("animal_model.class_ids must be valid COCO IDs from 0 through 79")
    selected_scores = scores[selected_ids]
    local_best = selected_scores.argmax(axis=0)
    confidences = selected_scores[local_best, np.arange(scores.shape[1])]
    predicted_ids = selected_ids[local_best]
    candidates = np.flatnonzero(confidences >= float(threshold))
    if not candidates.size:
        return []

    # Avoid decoding DFL distributions for all 8,400 anchors. Filtering scores
    # first usually leaves only a small subset and lowers host-side latency.
    all_box_logits = np.concatenate(box_heads, axis=1)
    if all_box_logits.shape[1] != ANCHORS.shape[1]:
        raise RuntimeError(f"YOLOv8n head has {all_box_logits.shape[1]} anchors; expected {ANCHORS.shape[1]}")
    dfl = all_box_logits[:, candidates].reshape(4, 16, -1).copy()
    dfl -= dfl.max(axis=1, keepdims=True)
    np.exp(dfl, out=dfl)
    dfl /= np.maximum(dfl.sum(axis=1, keepdims=True), 1e-12)
    distances = (dfl * DFL_BINS[None, :, None]).sum(axis=1)
    candidate_anchors = ANCHORS[:, candidates]
    candidate_strides = ANCHOR_STRIDES[candidates]
    model_xyxy = np.concatenate((candidate_anchors - distances[:2],
                                 candidate_anchors + distances[2:]), axis=0)
    model_xyxy *= candidate_strides[None, :]

    scale, pad_x, pad_y = letterbox_geometry(width, height)
    boxes = model_xyxy.copy()
    boxes[[0, 2]] = (boxes[[0, 2]] - pad_x) / scale
    boxes[[1, 3]] = (boxes[[1, 3]] - pad_y) / scale
    boxes[[0, 2]] = boxes[[0, 2]].clip(0, width)
    boxes[[1, 3]] = boxes[[1, 3]].clip(0, height)
    boxes = np.vstack((boxes, predicted_ids[candidates])).T
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, candidates = boxes[valid], candidates[valid]
    if not len(boxes):
        return []
    keep = _nms(boxes, confidences[candidates], iou_threshold, max_detections)
    return [{"box": tuple(float(v) for v in boxes[i, :4]),
             "class_id": int(boxes[i, 4]),
             "label": COCO_CLASSES[int(boxes[i, 4])],
             "score": float(confidences[candidates[i]])} for i in keep]


class AnimalDetector:
    """Own the compiled YOLOv8n model runner and its COCO head decoder."""

    def __init__(self, model_path: Path, seed_frame, pyneat, config: dict[str, Any]):
        options = pyneat.ModelOptions()
        options.preprocess.kind = pyneat.InputKind.Image
        options.preprocess.enable = pyneat.AutoFlag.On
        options.preprocess.color_convert.input_format = pyneat.PreprocessColorFormat.BGR
        options.preprocess.preset = pyneat.NormalizePreset.COCO_YOLO
        resize = options.preprocess.resize
        resize.enable = pyneat.AutoFlag.On
        resize.mode = pyneat.ResizeMode.Letterbox
        resize.width = INPUT_SIZE
        resize.height = INPUT_SIZE
        resize.pad_value = 114
        model = pyneat.Model(str(model_path), options)
        specs = model.output_specs()
        LOG.info("Loaded animal detector: %s; outputs: %s", model_path, specs)
        if len(specs) != 6:
            raise RuntimeError(f"YOLOv8n archive must expose six head outputs; found {len(specs)}")
        tensor = pyneat.Tensor.from_numpy(np.ascontiguousarray(seed_frame, dtype=np.uint8), copy=True,
                                          image_format=pyneat.PixelFormat.BGR,
                                          memory=pyneat.TensorMemory.EV74)
        run_options = pyneat.RunOptions()
        run_options.queue_depth = 1
        run_options.overflow_policy = pyneat.OverflowPolicy.Block
        run_options.preset = pyneat.RunPreset.Realtime
        self.runner = model.build([tensor], route_options=pyneat.ModelRouteOptions(), run_options=run_options)
        warmup = self.runner.run([tensor])
        if len(warmup) != 6:
            self.runner.close()
            raise RuntimeError(f"YOLOv8n warm-up returned {len(warmup)} outputs; expected 6")
        self.class_ids = frozenset(int(v) for v in config.get("class_ids", [0, 16, 18, 19, 21]))
        unknown = sorted(self.class_ids.difference(COCO_CLASSES))
        if unknown:
            self.runner.close()
            raise ValueError(f"Unknown COCO class IDs: {unknown}")
        self.threshold = float(config.get("score_threshold", 0.35))
        self.iou = float(config.get("nms_iou_threshold", 0.45))
        self.max_detections = int(config.get("max_detections", 30))
        self.pyneat = pyneat
        self.last_inference_ms = 0.0
        LOG.info("YOLOv8n active COCO classes: %s", [f"{i}:{COCO_CLASSES[i]}" for i in sorted(self.class_ids)])

    def infer(self, frame) -> list[dict[str, Any]]:
        tensor = self.pyneat.Tensor.from_numpy(np.ascontiguousarray(frame, dtype=np.uint8), copy=True,
                                               image_format=self.pyneat.PixelFormat.BGR,
                                               memory=self.pyneat.TensorMemory.EV74)
        started = time.perf_counter()
        outputs = self.runner.run([tensor])
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return decode_yolov8n(outputs, frame.shape[1], frame.shape[0], self.class_ids,
                              self.threshold, self.iou, self.max_detections)

    def close(self) -> None:
        self.runner.close()


def draw_detections(frame, detections: list[dict[str, Any]]):
    import cv2

    palette = {0: (255, 255, 0), 16: (0, 165, 255), 18: (0, 255, 255),
               19: (255, 0, 0), 21: (255, 0, 255)}
    height, width = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        class_id = det["class_id"]
        color = palette.get(class_id, (255, 255, 0))
        if det.get("intrusion", False):
            color = (0, 0, 255)
        p1 = (max(0, int(x1)), max(0, int(y1)))
        p2 = (min(width - 1, int(x2)), min(height - 1, int(y2)))
        cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
        prefix = "INTRUSION " if det.get("intrusion", False) else ""
        caption = f"{prefix}{det['label']} {det['score']:.2f}"
        cv2.putText(frame, caption, (p1[0], max(18, p1[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
