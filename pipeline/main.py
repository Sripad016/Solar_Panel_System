"""Live solar-panel segmentation: RTSP camera -> Modalix -> RTP/H.264 over UDP.

Run on the SiMa DevKit:
    python3 main.py --config config.yaml

The MPK is the surgically cut YOLOv8-seg model. Its 10 raw NHWC output tensors
are decoded on the host because this package does not contain the YOLO head.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from runtime_tuning import DetectionPolicy, LatencyWindow, resize_for_inference
from video_output import RtpVideoSender
from animal_detector import AnimalDetector, draw_detections
from output_metadata import OutputMetadata
from intrusion_zone import IntrusionZone
from rtsp_transport import configure_rtsp_transport

LOG = logging.getLogger("solar_panel_pipeline")
INPUT_SIZE = 640
PROTO_SIZE = 160
BOX_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]
STOP = threading.Event()


def request_stop(_signum: int, _frame: Any) -> None:
    STOP.set()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def resolve_path(value: str | Path, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def load_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 9:
        raise ValueError(f"This MPK has 9 classes; expected 9 labels in {path}, found {len(labels)}")
    return labels


def letterbox_geometry(width: int, height: int) -> tuple[float, float, float]:
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    return scale, (INPUT_SIZE - width * scale) / 2.0, (INPUT_SIZE - height * scale) / 2.0


def as_nchw(tensor: Any, channels: int, grid: int, name: str):
    """Convert an output tensor to [C,H,W], accepting only known MPK layouts."""
    import numpy as np

    # Consume views while `outputs` is alive; the next runner call happens only
    # after this decoder returns, so runtime-owned output buffers are not retained.
    array = np.asarray(tensor.to_numpy(copy=False), dtype=np.float32)
    if array.shape == (1, grid, grid, channels):
        return array[0].transpose(2, 0, 1)
    if array.shape == (1, channels, grid, grid):
        return array[0]
    raise RuntimeError(f"Unexpected {name} output shape {array.shape}; expected NHWC or NCHW for {grid}x{grid}x{channels}")


def build_anchors():
    import numpy as np

    anchor_points, strides = [], []
    for grid, stride in ((80, 8.0), (40, 16.0), (20, 32.0)):
        axis = np.arange(grid, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(axis, axis)
        anchor_points.append(np.stack((xx.ravel(), yy.ravel()), axis=0))
        strides.append(np.full(grid * grid, stride, dtype=np.float32))
    return np.concatenate(anchor_points, axis=1), np.concatenate(strides)


ANCHORS, STRIDES = build_anchors()
DFL_BINS = np.arange(16, dtype=np.float32)


def nms_by_class(boxes, scores, classes, max_detections: int,
                 iou_threshold: float, top_k_per_class: int):
    """Class-aware NMS with an OpenCV-version-independent implementation."""
    import cv2

    selected: list[int] = []
    for class_id in set(int(value) for value in classes):
        class_indices = [i for i, value in enumerate(classes) if int(value) == class_id]
        class_boxes = [boxes[i] for i in class_indices]
        class_scores = [float(scores[i]) for i in class_indices]
        kept = cv2.dnn.NMSBoxes(
            class_boxes, class_scores, 0.0, iou_threshold,
            top_k=max(1, int(top_k_per_class)),
        )
        if len(kept):
            selected.extend(class_indices[int(i)] for i in np.asarray(kept).reshape(-1))
    selected.sort(key=lambda index: float(scores[index]), reverse=True)
    return selected[:max_detections]


def decode_cut(outputs: list[Any], width: int, height: int, threshold: float,
               iou_threshold: float, max_detections: int,
               allowed_class_ids: frozenset[int],
               top_k_per_class: int = 50) -> list[dict[str, Any]]:
    """Decode the 10 raw outputs emitted by build_cut/best_cut."""
    import numpy as np

    if len(outputs) != 10:
        shapes = [tuple(getattr(tensor, "shape", ())) for tensor in outputs]
        raise RuntimeError(f"Expected 10 cut-model outputs; got {len(outputs)} with shapes {shapes}")

    grids = (80, 40, 20)
    box_parts, class_parts = [], []
    for scale_index, grid in enumerate(grids):
        box_parts.append(as_nchw(outputs[scale_index * 3], 64, grid, f"box[{scale_index}]").reshape(64, -1))
        class_parts.append(as_nchw(outputs[scale_index * 3 + 1], 9, grid, f"class[{scale_index}]").reshape(9, -1))
    class_logits = np.concatenate(class_parts, axis=1)
    if class_logits.shape[1] != ANCHORS.shape[1]:
        raise RuntimeError(f"Class head count {class_logits.shape[1]} does not match anchor count {ANCHORS.shape[1]}")

    # Filter by requested class scores first. Box DFL and mask coefficients are
    # only needed for survivors, avoiding full 8,400-anchor host postprocessing.
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(class_logits, -80.0, 80.0)))
    class_ids = probabilities.argmax(axis=0)
    confidences = probabilities.max(axis=0)
    class_allowed = np.isin(class_ids, tuple(allowed_class_ids))
    candidates = np.flatnonzero((confidences >= threshold) & class_allowed)
    if candidates.size == 0:
        return []

    box_logits = np.concatenate(box_parts, axis=1)
    if box_logits.shape[1] != ANCHORS.shape[1]:
        raise RuntimeError(f"Box head count {box_logits.shape[1]} does not match anchor count {ANCHORS.shape[1]}")
    distribution = box_logits[:, candidates].reshape(4, 16, -1).copy()
    distribution -= distribution.max(axis=1, keepdims=True)
    np.exp(distribution, out=distribution)
    distribution /= np.maximum(distribution.sum(axis=1, keepdims=True), 1e-12)
    distances = (distribution * DFL_BINS[None, :, None]).sum(axis=1)
    candidate_anchors = ANCHORS[:, candidates]
    candidate_strides = STRIDES[candidates]
    left_top = candidate_anchors - distances[:2]
    right_bottom = candidate_anchors + distances[2:]
    xyxy = np.concatenate((left_top, right_bottom), axis=0) * candidate_strides[None, :]

    x1, y1, x2, y2 = xyxy
    scale, pad_x, pad_y = letterbox_geometry(width, height)
    boxes = np.stack(((x1 - pad_x) / scale, (y1 - pad_y) / scale,
                      (x2 - pad_x) / scale, (y2 - pad_y) / scale), axis=1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    candidates, boxes, xyxy = candidates[valid], boxes[valid], xyxy[:, valid]
    if candidates.size == 0:
        return []

    nms_boxes = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    candidate_scores = confidences[candidates]
    candidate_classes = class_ids[candidates]
    keep = nms_by_class(
        nms_boxes, candidate_scores, candidate_classes, max_detections,
        iou_threshold, top_k_per_class,
    )

    coefficient_parts = [
        as_nchw(outputs[scale_index * 3 + 2], 32, grid,
                f"mask coefficient[{scale_index}]").reshape(32, -1)
        for scale_index, grid in enumerate(grids)
    ]
    coefficients = np.concatenate(coefficient_parts, axis=1)
    proto = as_nchw(outputs[9], 32, PROTO_SIZE, "mask prototype").reshape(32, -1)
    detections = []
    proto = proto.astype(np.float32, copy=False)
    for selected in keep:
        anchor_index = int(candidates[selected])
        mask_logits = coefficients[:, anchor_index] @ proto
        mask = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -80.0, 80.0)))
        bx1, by1, bx2, by2 = xyxy[:, selected].tolist()
        ox1, oy1, ox2, oy2 = boxes[selected].tolist()
        detections.append({
            "box": (ox1, oy1, ox2, oy2),
            "model_box": (bx1, by1, bx2, by2),
            "score": float(candidate_scores[selected]),
            "class_id": int(candidate_classes[selected]),
            "mask": mask.reshape(PROTO_SIZE, PROTO_SIZE),
        })
    return detections


def paint_detections(frame, detections: list[dict[str, Any]], labels: list[str],
                     alpha: float, mask_threshold: float) -> None:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale, pad_x, pad_y = letterbox_geometry(width, height)
    content_x1, content_y1 = int(round(pad_x)), int(round(pad_y))
    content_x2 = int(round(pad_x + width * scale))
    content_y2 = int(round(pad_y + height * scale))

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        bx1, by1, bx2, by2 = detection["model_box"]
        class_id = detection["class_id"]
        color = BOX_COLORS[class_id % len(BOX_COLORS)]

        # Restore the prototype mask at model resolution, remove letterbox padding,
        # then scale it into this detection's original-frame ROI.
        mx1 = max(0, content_x1, int(np.floor(bx1)))
        my1 = max(0, content_y1, int(np.floor(by1)))
        mx2 = min(INPUT_SIZE, content_x2, int(np.ceil(bx2)))
        my2 = min(INPUT_SIZE, content_y2, int(np.ceil(by2)))
        ox1, oy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        ox2, oy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if mx2 > mx1 and my2 > my1 and ox2 > ox1 and oy2 > oy1:
            mask_640 = cv2.resize(detection["mask"], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            mask_roi = mask_640[my1:my2, mx1:mx2]
            mask_roi = cv2.resize(mask_roi, (ox2 - ox1, oy2 - oy1), interpolation=cv2.INTER_LINEAR)
            visible = mask_roi >= mask_threshold
            region = frame[oy1:oy2, ox1:ox2]
            blended = (np.asarray(color, dtype=np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            region[visible] = blended[visible]

        px1 = min(width - 1, max(0, int(round(x1))))
        py1 = min(height - 1, max(0, int(round(y1))))
        px2, py2 = min(width - 1, int(round(x2))), min(height - 1, int(round(y2)))
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2, cv2.LINE_AA)
        label = labels[class_id] if 0 <= class_id < len(labels) else str(class_id)
        caption = f"{label} {detection['score']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, py1 - text_h - baseline - 4)
        cv2.rectangle(frame, (px1, top), (min(width - 1, px1 + text_w + 4), py1), color, -1)
        cv2.putText(frame, caption, (px1 + 2, max(text_h, py1 - baseline - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def make_input_tensor(frame, pyneat):
    import numpy as np

    return pyneat.Tensor.from_numpy(
        np.ascontiguousarray(frame, dtype=np.uint8), copy=True,
        image_format=pyneat.PixelFormat.BGR, memory=pyneat.TensorMemory.EV74,
    )


def build_runner(model_path: Path, seed_frame, pyneat):
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
    LOG.info("Loaded cut-model archive: %s", model_path)
    LOG.info("Model inputs: %s", model.input_specs())
    LOG.info("Model outputs: %s", model.output_specs())
    if len(model.output_specs()) != 10:
        raise RuntimeError(f"This pipeline expects the 10-output cut MPK; found {len(model.output_specs())} outputs")

    seed_tensor = make_input_tensor(seed_frame, pyneat)
    run_options = pyneat.RunOptions()
    run_options.queue_depth = 2
    run_options.overflow_policy = pyneat.OverflowPolicy.Block
    run_options.preset = pyneat.RunPreset.Realtime
    runner = model.build([seed_tensor], route_options=pyneat.ModelRouteOptions(), run_options=run_options)
    warmup = runner.run([seed_tensor])
    if len(warmup) != 10:
        runner.close()
        raise RuntimeError(f"Warm-up returned {len(warmup)} outputs; expected 10")
    LOG.info("Model warm-up complete")
    return runner


class LatestRtspReader:
    """Reconnectable RTSP reader that retains only the newest decoded frame."""

    def __init__(self, url: str, cv2):
        self.url = url
        self.cv2 = cv2
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0
        self.frame_read_at = 0.0
        self.connected = False
        self.thread = threading.Thread(target=self._read_loop, name="rtsp-reader", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read_loop(self) -> None:
        while not STOP.is_set():
            cap = self.cv2.VideoCapture(self.url, self.cv2.CAP_FFMPEG)
            cap.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                LOG.warning("Cannot open RTSP stream; retrying in 2 seconds: %s", self.url)
                cap.release()
                STOP.wait(2.0)
                continue
            self.connected = True
            LOG.info("RTSP connected: %s", self.url)
            while not STOP.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.frame_read_at = time.monotonic()
                    self.condition.notify_all()
            cap.release()
            self.connected = False
            if not STOP.is_set():
                LOG.warning("RTSP read stopped; reconnecting in 1 second")
                STOP.wait(1.0)

    def next_frame(self, after_sequence: int, timeout: float = 1.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.sequence <= after_sequence and not STOP.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, after_sequence, 0.0
                self.condition.wait(remaining)
            if STOP.is_set() or self.sequence <= after_sequence:
                return None, after_sequence, 0.0
            return self.frame, self.sequence, self.frame_read_at

    def close(self) -> None:
        STOP.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3.0)


def run_rtsp(cfg: dict[str, Any], config_dir: Path, labels: list[str], runner,
             animal_detector: AnimalDetector | None, pyneat, cv2) -> int:
    io_cfg = cfg["io"]
    transport = configure_rtsp_transport(io_cfg.get("rtsp_transport", "tcp"))
    LOG.info("Camera RTSP transport: %s", transport.upper())
    model_cfg = cfg["model"]
    decode_cfg = cfg["decode"]
    output_cfg = cfg["output"]
    reader = LatestRtspReader(str(io_cfg["rtsp_url"]), cv2)
    sender: RtpVideoSender | None = None
    max_fps = max(1.0, float(output_cfg.get("max_fps", 10)))
    min_interval = 1.0 / max_fps
    max_input_width = int(model_cfg.get("max_input_width", 1280))
    max_video_width = int(output_cfg.get("max_width", 960))
    max_det = int(decode_cfg.get("max_detections", 10))
    policy = DetectionPolicy.from_config(labels, decode_cfg)
    intrusion_zone = IntrusionZone(cfg.get("intrusion_zone", {}))
    metadata = OutputMetadata(output_cfg.get("metadata", {}))
    iou_threshold = float(decode_cfg.get("nms_iou_threshold", 0.45))
    alpha = float(output_cfg.get("mask_alpha", 0.5))
    mask_threshold = float(output_cfg.get("mask_threshold", 0.5))
    overlay = bool(output_cfg.get("overlay", True))
    snapshot_dir = resolve_path(io_cfg.get("snapshot_dir", "output"), config_dir)
    snapshot_interval = max(0, int(output_cfg.get("snapshot_interval_seconds", 30)))
    profiler = LatencyWindow(capacity=50)

    reader.start()
    LOG.info("RTP/H.264 output destination: %s:%d (max %.1f FPS)",
             output_cfg["udp_host"], int(output_cfg["udp_port"]), max_fps)
    LOG.info("Allowed classes: %s; score threshold: %.2f", sorted(policy.allowed_ids), policy.threshold)
    LOG.info("Camera: %s", io_cfg["rtsp_url"])
    if animal_detector is not None:
        LOG.info("Animal detector enabled; intrusion zone %s",
                 "configured" if intrusion_zone.is_configured else "empty (awaiting zone coordinates)")
    last_sequence = 0
    last_process_started = 0.0
    last_snapshot_at = 0.0
    processed = 0
    try:
        while not STOP.is_set():
            wait = min_interval - (time.monotonic() - last_process_started)
            if wait > 0:
                STOP.wait(wait)
                if STOP.is_set():
                    break

            frame, sequence, frame_read_at = reader.next_frame(last_sequence, timeout=1.0)
            if frame is None:
                continue
            last_sequence = sequence
            frame_started = time.perf_counter()
            last_process_started = time.monotonic()
            frame = resize_for_inference(frame, max_input_width, cv2)
            height, width = frame.shape[:2]
            tensor = make_input_tensor(frame, pyneat)
            model_started = time.perf_counter()
            outputs = runner.run([tensor])
            model_done = time.perf_counter()
            detections = decode_cut(
                outputs, width, height, policy.threshold, iou_threshold, max_det,
                policy.allowed_ids, int(decode_cfg.get("pre_nms_top_k_per_class", 50)),
            )
            decode_done = time.perf_counter()
            animal_started = time.perf_counter()
            animal_detections = animal_detector.infer(frame) if animal_detector is not None else []
            animal_done = time.perf_counter()
            intrusion_zone.update(animal_detections, width, height)
            if overlay:
                paint_detections(frame, detections, labels, alpha, mask_threshold)
                draw_detections(frame, animal_detections)
                intrusion_zone.draw(frame, cv2)
            combined_inference_ms = (model_done - model_started) * 1000.0
            if animal_detector is not None:
                combined_inference_ms += animal_detector.last_inference_ms
            metadata.draw(frame, combined_inference_ms, cv2)
            overlay_done = time.perf_counter()

            video_frame = resize_for_inference(frame, max_video_width, cv2)
            if sender is None:
                video_height, video_width = video_frame.shape[:2]
                sender = RtpVideoSender(
                    host=str(output_cfg["udp_host"]), port=int(output_cfg["udp_port"]),
                    width=video_width, height=video_height,
                    fps=int(round(max_fps)),
                    bitrate_kbps=int(output_cfg.get("bitrate_kbps", 1200)),
                    pyneat=pyneat, np=np,
                )
                LOG.info("Neat hardware H.264 sender ready on %s:%d (%dx%d@%d)",
                         output_cfg["udp_host"], sender.port, video_width, video_height, int(round(max_fps)))
            sender.send(video_frame)
            video_done = time.perf_counter()
            processed += 1
            completed_at = time.monotonic()
            profiler.add(
                frame_age=(completed_at - frame_read_at) * 1000.0,
                model=(model_done - model_started) * 1000.0,
                decode=(decode_done - model_done) * 1000.0,
                animal=(animal_done - animal_started) * 1000.0,
                overlay=(overlay_done - animal_done) * 1000.0,
                video_push=(video_done - overlay_done) * 1000.0,
                total=(video_done - frame_started) * 1000.0,
            )

            if snapshot_interval and completed_at - last_snapshot_at >= snapshot_interval:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snapshot_dir / "latest_rtsp.jpg"), frame)
                last_snapshot_at = completed_at
            if processed % 50 == 0:
                LOG.info("Frames=%d detections=%d latency p50/p95: %s",
                         processed, len(detections) + len(animal_detections), profiler.report())
    except KeyboardInterrupt:
        STOP.set()
    finally:
        reader.close()
        if sender is not None:
            sender.close()
        if animal_detector is not None:
            animal_detector.close()
        runner.close()
    LOG.info("Stopped. Processed %d frames.", processed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live solar panel segmentation on SiMa Modalix")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, request_stop)

    try:
        cfg = load_config(config_path)
        config_dir = config_path.parent
        model_path = resolve_path(cfg["model"]["path"], config_dir)
        labels_path = resolve_path(cfg["model"].get("labels", "labels.txt"), config_dir)
        if not model_path.is_file():
            raise FileNotFoundError(f"Compiled MPK archive not found: {model_path}")
        labels = load_labels(labels_path)

        import cv2
        import pyneat

        cv2.setNumThreads(1)
        cv2.setUseOptimized(True)
        seed = np.full((480, 640, 3), 114, dtype=np.uint8)
        runner = build_runner(model_path, seed, pyneat)
        animal_detector = None
        animal_cfg = cfg.get("animal_model", {})
        if animal_cfg.get("enabled", False):
            animal_path = resolve_path(animal_cfg["path"], config_dir)
            if not animal_path.is_file():
                runner.close()
                raise FileNotFoundError(f"Animal detector MPK archive not found: {animal_path}")
            animal_detector = AnimalDetector(animal_path, seed, pyneat, animal_cfg)
        return run_rtsp(cfg, config_dir, labels, runner, animal_detector, pyneat, cv2)
    except Exception as exc:
        LOG.exception("Pipeline startup/runtime failure: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
