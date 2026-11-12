"""Configure the camera's FFmpeg RTSP transport before opening the stream."""

import os


def configure_rtsp_transport(transport: str) -> str:
    transport = str(transport).strip().lower()
    if transport not in ("tcp", "udp"):
        raise ValueError("io.rtsp_transport must be tcp or udp")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"
    )
    return transport
