"""Inspectable runtime policy and timing helpers for the solar-panel app."""

from __future__ import annotations

from collections import defaultdict


DEFAULT_PANEL_CLASSES = (
    "Bird-drop",
    "Defective",
    "Dusty",
    "Electrical-Damage",
    "Non-Defective",
    "Physical-Damage",
    "Snow",
)


class DetectionPolicy:
    """Apply a closed-set class allowlist before box/mask postprocessing."""

    def __init__(self, labels: list[str], allowed_names: list[str], threshold: float):
        self.labels = labels
        self.threshold = float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("decode.score_threshold must be in [0, 1]")

        by_normalized_name = {name.strip().casefold(): i for i, name in enumerate(labels)}
        unknown = [name for name in allowed_names if name.strip().casefold() not in by_normalized_name]
        if unknown:
            raise ValueError(f"Unknown class names in decode.allowed_classes: {unknown}")
        self.allowed_ids = frozenset(by_normalized_name[name.strip().casefold()] for name in allowed_names)
        if not self.allowed_ids:
            raise ValueError("decode.allowed_classes must contain at least one model class")

    @classmethod
    def from_config(cls, labels: list[str], config: dict) -> "DetectionPolicy":
        names = config.get("allowed_classes", list(DEFAULT_PANEL_CLASSES))
        return cls(labels, list(names), float(config.get("score_threshold", 0.50)))

    def keep_class(self, class_id: int) -> bool:
        return class_id in self.allowed_ids


def resize_for_inference(frame, max_width: int, cv2):
    """Downscale large decoded frames before copying them into the model route."""
    width = int(frame.shape[1])
    if max_width <= 0 or width <= max_width:
        return frame
    output_width = max_width & ~1
    output_height = max(2, int(round(frame.shape[0] * output_width / width)) & ~1)
    return cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)


class LatencyWindow:
    """Collect a small rolling window and report median/p95 stage latency."""

    def __init__(self, capacity: int = 50):
        self.capacity = max(1, int(capacity))
        self.samples: dict[str, list[float]] = defaultdict(list)

    def add(self, **measurements_ms: float) -> None:
        for name, value in measurements_ms.items():
            values = self.samples[name]
            values.append(float(value))
            if len(values) > self.capacity:
                del values[0]

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
        return ordered[index]

    def report(self) -> str:
        fields = []
        for name in ("frame_age", "model", "decode", "animal", "overlay", "video_push", "total"):
            values = self.samples.get(name, [])
            if values:
                fields.append(f"{name}={self._percentile(values, 0.50):.1f}/{self._percentile(values, 0.95):.1f}ms")
        return " ".join(fields)
