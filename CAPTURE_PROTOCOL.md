# Drone Capture Protocol — Drone3D Research

Ship this to the collaborator flying the Pixhawk-based S500/X450 in India.

Goal: collect ~30 minutes of drone video across 4–6 short flights, with **synchronized Pixhawk telemetry**, suitable for 3D reconstruction research and self-supervised model adaptation.

---

## 1. Equipment checklist

- Drone: S500/X450 with Pixhawk (ArduPilot or PX4 — note which one, with firmware version).
- Camera: ideally a fixed RGB camera with **global shutter** (GoPro, Sony RX0, Arducam IMX477, etc.). Rolling shutter is acceptable but log the make/model. Avoid auto-focus and auto-exposure (lock both before flight if possible).
- SD cards (one for Pixhawk log, one for camera).
- Telemetry radio + ground station (Mission Planner / QGroundControl) for live monitoring.
- (Optional) Camera trigger wire from Pixhawk AUX out → camera trigger input, for frame-accurate timestamping. If not available, use **method B (LED sync)** below.

## 2. Pre-flight calibration (one time per session)

1. Calibrate accelerometer, gyro, compass in Mission Planner / QGC.
2. Confirm GPS lock with ≥10 satellites and HDOP <2.0.
3. Set logging mode in Pixhawk to **LOG_DISARMED=1** (ArduPilot) or equivalent so logs include pre-arm sensor data.
4. Set logging rate as high as the board allows (ArduPilot defaults are fine: GPS 5Hz, IMU 50Hz, attitude 50Hz, baro 10Hz).
5. Camera: lock white balance, lock exposure to a reasonable mid-bright value, lock focus to infinity (or hyperfocal). Disable any image stabilization that crops the frame variably.
6. Format SD cards (camera and Pixhawk) to start each session with a clean log.

## 3. Time synchronization — pick ONE method

### Method A (preferred): Pixhawk camera trigger
- Wire Pixhawk AUX channel to camera trigger.
- In ArduPilot: `CAM_TRIGG_TYPE=1` (servo), set `CAM_SERVO_ON/OFF` for your camera.
- Pixhawk logs CAM messages with precise timestamps each time it triggers. Use these as the per-frame ground-truth time index.

### Method B (fallback): LED + clap sync
- Mount a bright LED on the drone, visible in the camera frame.
- At start of flight, flash the LED 3 times by toggling a Pixhawk AUX servo (or manually pressing a button wired to the LED).
- The Pixhawk log records the AUX servo toggle timestamps; the camera records the LED visually. Cross-correlate offline to find the time offset.
- Repeat at end of flight to detect any clock drift.

### Method C (last resort): GPS-UTC alignment
- Both Pixhawk (via GPS) and most cameras (via on-board RTC) record UTC. Use them to align with ~0.5s accuracy. Worse than A/B but works.

**Record which method you used in the flight log notes.**

## 4. Flight patterns (do all four per session)

Each pattern at **3 altitudes** if airspace allows: ~30 m, ~60 m, ~100 m AGL.

| # | Pattern | Duration | Purpose |
|---|---|---|---|
| 1 | **Orbital** — circle a single building/landmark at fixed radius (~30–50 m), camera pointed inward and tilted ~30° down | 60–90 s | Dense multi-view for static scene reconstruction |
| 2 | **Lawnmower / boustrophedon** — straight passes back and forth over an area, camera nadir (straight down) | 90–120 s | Mapping / DSM-style coverage |
| 3 | **Vertical sweep** — ascend slowly from 10 m to 100 m while orbiting slowly, camera tilted ~30° down | 60 s | Altitude generalization, scale variation |
| 4 | **Free-form translation** — fly forward in a straight line ~50 m at constant altitude, then sideways ~50 m, then forward — camera locked forward | 60 s | Translation-dominant motion (cleanest signal for photometric loss) |

**Avoid:** pure hover (no parallax), aggressive yaw (rolling shutter wreckage), flight against the sun (lens flare kills photometric loss), and very fast translation (motion blur).

## 5. Scene diversity — capture across these scene types (one session each)

1. **Dense urban** — buildings, varied heights, some vehicles/pedestrians (we *want* dynamic distractors for the masking experiments).
2. **Vegetation / mixed** — trees, fields, a few buildings.
3. **Open ground** — parking lot, large flat area with structures at the edges. Tests low-texture regions.
4. **High-contrast / repetitive** — rooftops, regular patterns. Tests the model's failure modes.

## 6. Camera settings

- Resolution: ≥ 1080p, ideally 4K downsampled to 1080p afterward (sharper).
- Frame rate: 30 fps preferred (gives flexibility to subsample). 60 fps if storage allows.
- Shutter speed: 1/(2× frame rate) or faster. For 30fps → 1/60s or faster.
- ISO: lowest possible while keeping shutter fast. Avoid auto-ISO.
- File format: MP4 H.264/H.265 at high bitrate (≥ 50 Mbps).

## 7. Per-flight notes file

For each flight, save a `flight_NN.md` next to the video with:

```
Date/time (UTC):
Location (lat, lon, rough name):
Weather (cloud cover %, wind speed, time of day):
Drone firmware (ArduPilot/PX4 + version):
Camera model + settings (resolution, fps, ISO, shutter):
Sync method (A/B/C):
Pattern executed: (1/2/3/4)
Altitude(s):
Anything unusual: (e.g., gusts, camera FOV changes, GPS dropouts)
```

## 8. After each session — package and upload

Bundle into `flight_YYYYMMDD_HHMM/`:
```
flight_YYYYMMDD_HHMM/
├── video.mp4
├── pixhawk.bin           (or .ulg for PX4)
├── pixhawk.csv           (telemetry exported — see step 9)
├── notes.md              (template above)
└── thumbnails/           (10 sample frames, optional but helpful)
```

Upload to a shared Google Drive / rsync target.

## 9. Telemetry export (one quick step on the collaborator's machine)

For ArduPilot (.bin):
```bash
pip install pymavlink
mavlogdump.py --types GPS,IMU,ATT,BARO,RATE,CAM --format csv pixhawk.bin > pixhawk.csv
```

For PX4 (.ulg):
```bash
pip install pyulog
ulog2csv pixhawk.ulg
```

This produces a CSV the analysis pipeline can consume directly. Verify the file has > 0 rows for `GPS`, `IMU`, `ATT`, `BARO` before uploading.

## 10. First-pass quality gate (collaborator runs this before uploading)

For each captured flight:
1. Play the video — confirm focus, exposure, no major prop-in-shot.
2. Open the .bin log in Mission Planner — confirm GPS has lock throughout, IMU not saturating, no error messages.
3. Eyeball: video length ≈ Pixhawk log length (within a few seconds — armed-to-disarmed time should match).
4. If sync method B: confirm LED flashes are visible in the first and last seconds of video.

If any check fails, redo the flight rather than uploading bad data — bad telemetry sync silently corrupts the entire research signal.

---

## First-batch ask

For the first batch, do **one flight of Pattern 1 (orbital)** over a single static building at ~50 m, plus the telemetry CSV. Just one flight, ~90 seconds of video. We use that to debug the full processing pipeline before asking for the full set.
