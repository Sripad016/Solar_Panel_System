"""Virtual-zone intrusion evaluation and drawing for live detections."""

from __future__ import annotations

import logging
import time
import numpy as np

from animal_detector import COCO_CLASSES

LOG = logging.getLogger("solar_panel_pipeline.intrusion")


class IntrusionZone:
    """Mark configured person/animal detections inside a normalized polygon."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.polygon = [(float(point[0]), float(point[1]))
                        for point in config.get("zone_polygon_normalized", [])]
        self.margin_pixels = max(0, int(config.get("zone_margin_pixels", 0)))
        if self.polygon and len(self.polygon) < 3:
            raise ValueError("intrusion_zone.zone_polygon_normalized needs at least 3 [x,y] points")
        if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in self.polygon):
            raise ValueError("intrusion-zone coordinates must be normalized to [0,1]")
        self.class_ids = frozenset(int(value) for value in config.get("class_ids", [0, 16, 18, 19, 21]))
        unknown = self.class_ids.difference(COCO_CLASSES)
        if unknown:
            raise ValueError(f"Unknown COCO IDs in intrusion zone: {sorted(unknown)}")
        self.was_occupied = False
        self.last_alert_at = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self.polygon) or self.margin_pixels > 0

    def _polygon_for_frame(self, width: int, height: int) -> list[tuple[float, float]]:
        if self.polygon:
            return self.polygon
        margin = self.margin_pixels
        if margin <= 0 or width <= 2 * margin or height <= 2 * margin:
            return []
        return [(margin / width, margin / height),
                (1.0 - margin / width, margin / height),
                (1.0 - margin / width, 1.0 - margin / height),
                (margin / width, 1.0 - margin / height)]

    @staticmethod
    def _contains(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        inside = False
        previous = len(polygon) - 1
        for current, (x1, y1) in enumerate(polygon):
            x0, y0 = polygon[previous]
            if (y1 > y) != (y0 > y):
                crossing_x = (x0 - x1) * (y - y1) / (y0 - y1) + x1
                if x < crossing_x:
                    inside = not inside
            previous = current
        return inside

    def update(self, detections: list[dict], width: int, height: int) -> int:
        """Tag inside detections, log one message on zone-entry transition."""
        inside_labels: set[str] = set()
        polygon = self._polygon_for_frame(width, height)
        if polygon:
            for detection in detections:
                detection["intrusion"] = False
                class_id = int(detection.get("class_id", -1))
                if class_id not in self.class_ids:
                    continue
                x1, y1, x2, y2 = detection["box"]
                center = ((x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height))
                if self._contains(center, polygon):
                    detection["intrusion"] = True
                    inside_labels.add(COCO_CLASSES[class_id])

        occupied = bool(inside_labels)
        now = time.monotonic()
        if occupied and not self.was_occupied and now - self.last_alert_at >= 5.0:
            LOG.warning("[INTRUSION] Zone entered by: %s", ", ".join(sorted(inside_labels)))
            self.last_alert_at = now
        self.was_occupied = occupied
        return sum(bool(detection.get("intrusion")) for detection in detections)

    def draw(self, frame, cv2) -> None:
        height, width = frame.shape[:2]
        polygon = self._polygon_for_frame(width, height)
        if not polygon:
            cv2.putText(frame, "INTRUSION ZONE NOT CONFIGURED", (10, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 2, cv2.LINE_AA)
            return
        points = [(round(x * width), round(y * height)) for x, y in polygon]
        cv2.polylines(frame, [np.asarray(points, dtype="int32")],
                      True, (0, 255, 255), 2, cv2.LINE_AA)
        label_x = max(4, min(width - 4, points[0][0]))
        label_y = max(20, min(height - 4, points[0][1] - 7))
        cv2.putText(frame, "VIRTUAL INTRUSION ZONE", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
