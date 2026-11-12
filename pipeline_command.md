# Pipeline Commands

Project network: camera `192.168.1.10:5005/routecam`, SiMa DevKit `192.168.1.20`, laptop `192.168.1.5`. The video output is RTP/H.264 over UDP port `5000`.

## Start the GStreamer camera viewer on the laptop

Run from the project directory and leave this terminal open:

```bash
cd /workspace/Solar_Panel_POC
python3 pipeline/viewer.py --port 5000 --jitter-ms 10
```

Direct GStreamer command if you prefer not to use `viewer.py`:

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

## Run the pipeline on the DevKit

Open a second laptop terminal, then connect to the DevKit:

```bash
ssh sima@192.168.1.20
cd ~/solar_panel_poc/pipeline
source ~/pyneat/bin/activate
python3 -u main.py --config config.yaml
```

Keep this terminal open. Stop the foreground pipeline with `Ctrl+C`.

## Run the pipeline in the background

From an SSH session on the DevKit:

```bash
cd ~/solar_panel_poc/pipeline
source ~/pyneat/bin/activate
nohup python3 -u main.py --config config.yaml > run.log 2>&1 < /dev/null &
echo $! > pipeline.pid
```

View its log:

```bash
tail -f ~/solar_panel_poc/pipeline/run.log
```

## Stop the laptop camera viewer

For a foreground viewer, press `Ctrl+C` in its terminal.

To stop the GStreamer receiver by its UDP port from another laptop terminal:

```bash
pgrep -af '[g]st-launch-1.0.*port=5000'
pkill -TERM -f '[g]st-launch-1.0.*port=5000'
```

If it was started through `viewer.py`, stop the Python wrapper too:

```bash
pkill -TERM -f '[p]ython3 pipeline/viewer.py --port 5000'
```

Check that it stopped:

```bash
pgrep -af '[g]st-launch-1.0.*port=5000' || echo 'GStreamer stopped'
pgrep -af '[p]ython3 pipeline/viewer.py --port 5000' || echo 'Viewer stopped'
```

## Stop the DevKit pipeline

If it is running in the foreground, press `Ctrl+C` in the SSH terminal. To stop a background run from the laptop:

```bash
ssh sima@192.168.1.20 'cd ~/solar_panel_poc/pipeline && if [ -f pipeline.pid ]; then kill -CONT "$(cat pipeline.pid)" 2>/dev/null || true; kill -TERM "$(cat pipeline.pid)"; else pkill -TERM -f "^python3( -u)? main.py --config config.yaml$"; fi'
```

The `CONT` handles a process suspended with Ctrl+Z. Confirm the app has exited:

```bash
ssh sima@192.168.1.20 'pgrep -af "^python3( -u)? main.py --config config.yaml$" || echo "Pipeline stopped"'
```

If it remains after a few seconds, inspect it before force-stopping:

```bash
ssh sima@192.168.1.20 'ps -fp "$(cat ~/solar_panel_poc/pipeline/pipeline.pid)"'
ssh sima@192.168.1.20 'kill -KILL "$(cat ~/solar_panel_poc/pipeline/pipeline.pid)"'
```
