"""Low-latency H.264/RTP video output using Neat's hardware VideoSender."""

from __future__ import annotations


class RtpVideoSender:
    """Push annotated BGR frames through the public PyNeat VideoSender graph."""

    def __init__(self, host: str, port: int, width: int, height: int,
                 fps: int, bitrate_kbps: int, pyneat, np):
        options = pyneat.InputOptions()
        options.payload_type = pyneat.PayloadType.Image
        options.format = pyneat.Format.BGR
        options.width = int(width)
        options.height = int(height)
        options.depth = 3
        options.fps_n = max(1, int(fps))
        options.fps_d = 1

        encode = pyneat.SimaEncodeOptions()
        encode.type = pyneat.SimaEncodeType.H264
        encode.fps = max(1, int(fps))
        encode.gop_length = max(1, int(fps))
        sender_options = pyneat.VideoSenderOptions.from_raw(encode)
        sender_options.host = str(host)
        sender_options.channel = 0
        sender_options.video_port_base = int(port)
        sender_options.encoder.bitrate_kbps = int(bitrate_kbps)
        sender_options.async_ = False
        self.port = sender_options.video_port

        self.graph = pyneat.Graph("solar_panel_rtp_video")
        self.graph.add(pyneat.nodes.input(options))
        self.graph.add(pyneat.groups.video_sender(sender_options))

        seed = pyneat.Tensor.from_numpy(
            np.zeros((height, width, 3), dtype=np.uint8),
            copy=True,
            image_format=pyneat.PixelFormat.BGR,
            memory=pyneat.TensorMemory.CPU,
        )
        run_options = pyneat.RunOptions()
        run_options.queue_depth = 1
        run_options.overflow_policy = pyneat.OverflowPolicy.KeepLatest
        run_options.preset = pyneat.RunPreset.Realtime
        self.run = self.graph.build([seed], options=run_options)
        self.pyneat = pyneat
        self.np = np

    def send(self, frame) -> None:
        tensor = self.pyneat.Tensor.from_numpy(
            self.np.ascontiguousarray(frame, dtype=self.np.uint8),
            copy=True,
            image_format=self.pyneat.PixelFormat.BGR,
            memory=self.pyneat.TensorMemory.CPU,
        )
        if not self.run.push([tensor]):
            raise RuntimeError("Neat VideoSender rejected an annotated frame")

    def close(self) -> None:
        self.run.close()
