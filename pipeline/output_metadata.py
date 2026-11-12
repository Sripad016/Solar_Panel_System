"""Draw live pipeline performance and time metadata on output frames."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import time


class OutputMetadata:
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.clock_offset_seconds = float(config.get("timestamp_utc_offset_seconds", 0.0))
        self.history = deque(maxlen=max(2, int(config.get("fps_window_frames", 30))))

    def draw(self, frame, inference_ms: float, cv2) -> None:
        if not self.enabled:
            return
        height, width = frame.shape[:2]
        now_mono = time.monotonic()
        self.history.append(now_mono)
        fps = 0.0
        if len(self.history) > 1:
            elapsed = self.history[-1] - self.history[0]
            if elapsed > 0:
                fps = (len(self.history) - 1) / elapsed

        now_utc = datetime.now(timezone.utc) + timedelta(seconds=self.clock_offset_seconds)
        now_ist = now_utc.astimezone(timezone(timedelta(hours=5, minutes=30), name="IST"))
        lines = [
            f"FPS: {fps:.1f}",
            f"Inference: {max(0.0, float(inference_ms)):.1f} ms",
            f"Timestamp: {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}",
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        line_height = 23
        padding = 9
        max_text_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
        panel_w = min(width - 12, max_text_width + padding * 2)
        panel_h = line_height * len(lines) + padding * 2
        x0, y0 = 8, 8
        x1, y1 = min(width - 1, x0 + panel_w), min(height - 1, y0 + panel_h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (18, 18, 18), -1)
        for index, line in enumerate(lines):
            baseline_y = y0 + padding + (index + 1) * line_height - 4
            cv2.putText(frame, line, (x0 + padding, baseline_y), font, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
