"""Launch a GStreamer RTP/H.264 receiver on the laptop.

Run on the laptop after starting main.py on the DevKit:
    python3 viewer.py --port 5000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="View the Modalix H.264/RTP UDP stream with GStreamer")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--jitter-ms", type=int, default=10)
    args = parser.parse_args()
    gst = shutil.which("gst-launch-1.0")
    if not gst:
        print("gst-launch-1.0 is not installed or not on PATH", file=sys.stderr)
        return 2

    rtp_caps = (
        "application/x-rtp,media=video,clock-rate=90000,"
        "encoding-name=H264,payload=96"
    )
    command = [
        gst, "-v", "udpsrc", f"port={args.port}", f"caps={rtp_caps}",
        "!", "rtpjitterbuffer", f"latency={max(0, args.jitter_ms)}", "drop-on-latency=true",
        "!", "rtph264depay", "!", "decodebin",
        "!", "queue", "leaky=downstream", "max-size-buffers=1", "max-size-bytes=0", "max-size-time=0",
        "!", "videoconvert",
        "!", "autovideosink", "sync=false",
    ]
    print("Receiving RTP/H.264 on UDP port", args.port, "— press Ctrl+C to stop", flush=True)
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
