# Solar Panel POC: End-to-End Runbook

This guide starts the RTSP camera → SiMa DevKit → laptop GStreamer video pipeline from a Linux laptop. It covers setup, transfer, start, logs, stop, and common recovery steps.

## Project addresses and current settings

| Device | Address / service |
|---|---|
| Camera | `192.168.1.10:5005`, RTSP path `/routecam` |
| SiMa DevKit | `192.168.1.20`, SSH user `sima` |
| Laptop | `192.168.1.5` |
| Laptop video receiver | UDP port `5000` |

The current configuration uses RTSP over UDP from the camera, RTP/H.264 over UDP to the laptop, 1280×720 output, a 5000 kbps encoder bitrate, and a 30 FPS cap. The solar detector uses confidence `0.50` and up to 10 detections. The second YOLOv8n detector checks person, dog, sheep, cow, and bear. The virtual zone is the full image inset by 60 pixels on each side.

## 1. Prepare the laptop

Connect the laptop, camera, and DevKit to the same reachable network. Set the laptop address to `192.168.1.5` and make sure it can reach the other devices:

```bash
ping -c 3 192.168.1.20
ping -c 3 192.168.1.10
```

Install GStreamer if it is not already installed:

```bash
sudo apt update
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-libav
```

Check that the receiver elements are available:

```bash
gst-inspect-1.0 udpsrc
gst-inspect-1.0 rtpjitterbuffer
gst-inspect-1.0 rtph264depay
gst-inspect-1.0 decodebin
```

If the laptop firewall is enabled, allow the video port:

```bash
sudo ufw allow 5000/udp
```

## 2. Check the project files

Run these commands from the project root on the laptop. If you copied the project somewhere else, replace `/workspace/Solar_Panel_POC` in the commands below with that folder:

```bash
cd /workspace/Solar_Panel_POC
test -f pipeline/config.yaml
test -f pipeline/main.py
test -f build_cut/best_cut/best_cut_mpk.tar.gz
test -f yolo_v8n_mpk.tar.gz
```

The solar model and detector archive must remain at the paths referenced by `pipeline/config.yaml`:

```text
Solar_Panel_POC/
├── build_cut/best_cut/best_cut_mpk.tar.gz
├── yolo_v8n_mpk.tar.gz
└── pipeline/
    ├── config.yaml
    ├── main.py
    ├── animal_detector.py
    ├── intrusion_zone.py
    ├── output_metadata.py
    ├── rtsp_transport.py
    └── video_output.py
```

## 3. Configure the pipeline

Open `pipeline/config.yaml` and check these values before each run:

```yaml
io:
  rtsp_url: rtsp://192.168.1.10:5005/routecam
  rtsp_transport: udp

output:
  udp_host: 192.168.1.5
  udp_port: 5000
  max_fps: 30
  max_width: 1280
  bitrate_kbps: 5000
```

The UDP camera transport has been tested with this camera. If the RTSP reader repeatedly disconnects or the picture shows packet-loss artifacts, change `rtsp_transport: udp` to `rtsp_transport: tcp` and restart the application.

The solar settings are currently `decode.score_threshold: 0.50` and `decode.max_detections: 10`. Animal classes are configured under `animal_model.class_ids: [0, 16, 18, 19, 21]`. The zone uses `intrusion_zone.zone_margin_pixels: 60`; boxes for people or those animals inside it turn red and log an intrusion alert.

The timestamp overlay has a fixed correction because the DevKit clock was not synchronized. After synchronizing the DevKit clock with NTP, set `output.metadata.timestamp_utc_offset_seconds` to `0`.

## 4. Copy the app and models to the DevKit

The DevKit needs its existing SiMa SDK and Python environment at `~/pyneat`. Copy the pipeline and both MPKs from the project root:

```bash
cd /workspace/Solar_Panel_POC
ssh sima@192.168.1.20 'mkdir -p ~/solar_panel_poc/build_cut/best_cut'
scp -r pipeline sima@192.168.1.20:~/solar_panel_poc/
scp build_cut/best_cut/best_cut_mpk.tar.gz \
  sima@192.168.1.20:~/solar_panel_poc/build_cut/best_cut/
scp yolo_v8n_mpk.tar.gz sima@192.168.1.20:~/solar_panel_poc/
```

SSH will prompt for the DevKit account credentials unless SSH keys are configured. Do not put the password directly into a command.

## 5. Start the laptop GStreamer viewer

Start this first in a laptop terminal and leave it open:

```bash
cd /workspace/Solar_Panel_POC
python3 pipeline/viewer.py --port 5000 --jitter-ms 10
```

The viewer includes a one-frame leaky queue after decoding so the display favors the newest decoded frame if rendering falls behind. You can also run the receiver directly:

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

Stop a foreground viewer with `Ctrl+C` in its terminal. To stop it from another laptop terminal, run:

```bash
pgrep -af '[g]st-launch-1.0.*port=5000'
pkill -TERM -f '[g]st-launch-1.0.*port=5000'
pkill -TERM -f '[p]ython3 pipeline/viewer.py --port 5000'
```

Confirm the processes have stopped:

```bash
pgrep -af '[g]st-launch-1.0.*port=5000' || echo 'GStreamer receiver stopped'
pgrep -af '[p]ython3 pipeline/viewer.py --port 5000' || echo 'Python viewer stopped'
```

## 6. Start the DevKit pipeline

For an interactive run, open another laptop terminal:

```bash
ssh sima@192.168.1.20
cd ~/solar_panel_poc/pipeline
source ~/pyneat/bin/activate
python3 -u main.py --config config.yaml
```

Keep this terminal open. The app should log that both models loaded, RTSP connected, and the H.264 sender is ready at `192.168.1.5:5000`. Stop an interactive run with `Ctrl+C`.

To run it in the background instead, use an SSH session and run:

```bash
cd ~/solar_panel_poc/pipeline
source ~/pyneat/bin/activate
nohup python3 -u main.py --config config.yaml > run.log 2>&1 < /dev/null &
echo $! > pipeline.pid
```

## 7. Check status and logs

From the DevKit SSH session:

```bash
pgrep -af '^python3( -u)? main.py --config config.yaml$'
tail -f ~/solar_panel_poc/pipeline/run.log
```

The output overlay shows processed FPS, model inference time, and an IST timestamp. The DevKit log reports rolling p50/p95 timing for frame age, solar model, solar decode, animal stage, overlays, video send, and total processing. A recent configuration has processed around 29 FPS, but the actual rate depends on scene complexity and system load.

To inspect the latest annotated frame on the DevKit:

```bash
ls -lh ~/solar_panel_poc/pipeline/output/latest_rtsp.jpg
```

## 8. Stop or restart the DevKit app

If the app is in the foreground, press `Ctrl+C` in that SSH terminal.

If it was started in the background, stop it from the laptop with this SSH command:

```bash
ssh sima@192.168.1.20 'pkill -TERM -f "^python3( -u)? main.py --config config.yaml$"'
```

Check that it stopped:

```bash
ssh sima@192.168.1.20 'pgrep -af "^python3( -u)? main.py --config config.yaml$" || echo "Pipeline stopped"'
```

If a process remains after several seconds, inspect it before force-stopping:

```bash
ssh sima@192.168.1.20 'ps -fp "$(cat ~/solar_panel_poc/pipeline/pipeline.pid)"'
ssh sima@192.168.1.20 'kill -KILL "$(cat ~/solar_panel_poc/pipeline/pipeline.pid)"'
```

To restart, run the background start commands in section 6 again. The run log is overwritten when starting with the supplied `>` redirection. To preserve earlier logs, change it to `>> run.log`.

## 9. Common recovery steps

**No window or no UDP video:** confirm the GStreamer viewer is running first, the laptop is `192.168.1.5`, and UDP port 5000 is allowed through the laptop firewall. Check the DevKit log for `hardware H.264 sender ready`.

**RTSP will not connect over UDP:** change `io.rtsp_transport` to `tcp`, copy the updated config to the DevKit, and restart the app. Confirm the camera URL is `rtsp://192.168.1.10:5005/routecam` and both devices can reach the camera.

**A model archive is missing:** repeat the `scp` commands in section 4 and confirm the archive paths in the project tree. The config resolves model paths relative to `pipeline/config.yaml`.

**The video freezes or accumulates delay:** inspect the logged `frame_age` and `total` timings. Restart the laptop viewer to clear its receive/decode queue. The viewer queue may drop old frames under overload to keep the display current.

**Change output or detection settings:** edit `pipeline/config.yaml`, copy it to the DevKit with `scp pipeline/config.yaml sima@192.168.1.20:~/solar_panel_poc/pipeline/`, then restart the app.
