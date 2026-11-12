# Solar Panel Live Segmentation Pipeline

The app receives the camera's RTSP stream, runs the cut YOLOv8 segmentation MPK on the SiMa Modalix DevKit, draws masks and labels, encodes the annotated frames with Neat's hardware H.264 sender, and sends standard RTP packets over UDP.

| Component | Address / path |
|---|---|
| SiMa DevKit | `192.168.1.20` |
| Camera | `rtsp://192.168.1.10:5005/routecam` |
| Laptop RTP/H.264 receiver | `192.168.1.5:5000/UDP` |
| Compiled model | `../build_cut/best_cut/best_cut_mpk.tar.gz` (relative to this folder) |
| COCO detector | `../yolo_v8n_mpk.tar.gz` (relative to this folder) |

The existing nine class labels are retained in `labels.txt`. Their order must match the model training order. A second detection-only `yolo_v8n_mpk.tar.gz` runs on the same camera frames and draws boxes for COCO `person`, `dog`, `sheep`, `cow`, and `bear`.

## Transfer runtime files to the DevKit

Run from the project root on the laptop:

```bash
ssh sima@192.168.1.20 'mkdir -p ~/solar_panel_poc/build_cut/best_cut'
scp -r pipeline sima@192.168.1.20:~/solar_panel_poc/
scp build_cut/best_cut/best_cut_mpk.tar.gz sima@192.168.1.20:~/solar_panel_poc/build_cut/best_cut/
scp yolo_v8n_mpk.tar.gz sima@192.168.1.20:~/solar_panel_poc/
```

## View with GStreamer on the laptop

Start the receiver before the DevKit app. The laptop needs GStreamer with RTP/H.264 depayloading, an H.264 decoder, and a video sink. Run:

```bash
gst-launch-1.0 -v \
  udpsrc port=5000 caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" \
  ! rtpjitterbuffer latency=10 drop-on-latency=true \
  ! rtph264depay \
  ! decodebin \
  ! queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 \
  ! videoconvert \
  ! autovideosink sync=false
```

Or run `python3 viewer.py --port 5000`, which launches the same receiver pipeline. Allow inbound UDP port `5000` through the laptop firewall.

## Run on the DevKit

```bash
ssh sima@192.168.1.20
cd ~/solar_panel_poc/pipeline
source ~/pyneat/bin/activate
python3 main.py --config config.yaml
```

Stop with `Ctrl+C`. The app handles `SIGTERM`, reconnects after RTSP interruptions, retains only the newest camera frame, and writes `output/latest_rtsp.jpg` snapshots every 30 seconds.

## Configuration and model contract

Edit `config.yaml` to change the camera URL, laptop address/port, thresholds, allowed classes, maximum inference width, output frame rate/width, or H.264 bitrate. The solar model is filtered to its panel classes. The separate COCO model uses class IDs `0, 16, 18, 19, 21` (`person`, `dog`, `sheep`, `cow`, `bear`) for bounding-box detections. Its decoder is in `animal_detector.py` for review.

`io.rtsp_transport` selects `udp` or `tcp` for the camera connection; restart the DevKit application after changing it. The laptop output remains RTP over UDP.

The solar MPK exposes 10 outputs, including three YOLOv8 segmentation heads and mask prototypes. The COCO detector exposes six outputs (three DFL box heads and three 80-class score heads). Both run on each sampled camera frame. The app downsizes large frames to 1280 pixels, outputs up to 1280 pixels and 30 FPS, and uses Neat's hardware encoder at the configured bitrate. Rolling p50/p95 timings include both model paths.

The virtual intrusion-zone logic is in `intrusion_zone.py`. The current zone uses `intrusion_zone.zone_margin_pixels: 20`, making it the full frame inset by 20 pixels on every side. Person, dog, sheep, cow, and bear are checked. A detection whose box center enters the zone is logged and its box and label turn red; the zone is outlined in yellow. A normalized polygon can replace the margin-based zone when a more specific boundary is needed.

The outgoing video includes an on-frame overlay with rolling processed FPS, combined solar and COCO model execution time, and a UTC timestamp. The DevKit clock was not NTP-synchronized, so `output.metadata.timestamp_utc_offset_seconds` corrects its current drift. Set that value to `0` after synchronizing the DevKit clock.

The DevKit requires SiMa PyNeat in `~/pyneat`, OpenCV with FFmpeg RTSP support, and NumPy/PyYAML. The laptop needs the GStreamer RTP/H.264 receiver elements and a display sink.
