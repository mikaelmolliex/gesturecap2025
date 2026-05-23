# Quickstart

This guide gets you from a blank machine to streaming hand-landmark data over OSC in a few minutes.

---

## 1. Install uv

`uv` is a fast Python package manager. Run the one-liner for your OS:

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After install, restart your terminal (or `source ~/.bashrc` / `source ~/.zshrc`) so the `uv` command is on your PATH.

---

## 2. Clone the repo

```bash
git clone https://github.com/m2b3/gesturecap2025.git
cd gesturecap2025
```

---

## 3. Install dependencies

```bash
uv sync
```

This creates a `.venv` folder and installs everything (MediaPipe, OpenCV, python-osc). No manual `pip` or virtual-env steps needed.

---

## 4. Hand model

The model file (`models/hand_landmarker.task`) is bundled with the repo, so you should already have it after cloning.

If it's missing for any reason, download it manually:

```bash
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

## 5. Set your OSC target

By default the script sends OSC to `127.0.0.1:11111` — the same machine, port 11111.  
If your audio software (Pure Data, Max/MSP, SuperCollider, etc.) listens on a different address or port, edit the two lines near the top of [doublehand_mp.py](doublehand_mp.py):

```python
OSC_IP   = "127.0.0.1"   # ← change to your machine's IP if needed
OSC_PORT = 11111          # ← change to match your patch's inlet port
```

---

## 6. Run

```bash
uv run python doublehand_mp.py
```

A preview window opens showing your webcam feed with hand skeletons drawn on it.  
Press **`q`** in the preview window to quit cleanly.

---

## What gets sent

For every frame, each visible hand emits **63 OSC messages** — one per axis per MediaPipe joint, each carrying a single float:

```
/left_wrist_x              <float>   # 0.0 = left edge,  1.0 = right edge
/left_wrist_y              <float>   # 0.0 = top edge,   1.0 = bottom edge
/left_wrist_z              <float>   # depth, relative to wrist (≈0 at wrist)
/left_thumb_cmc_x          <float>
/left_thumb_cmc_y          <float>
/left_thumb_cmc_z          <float>
...
/right_pinky_tip_z         <float>
```

The 21 joint names follow MediaPipe's `HandLandmark` enum (lowercased):

```
wrist,
thumb_cmc, thumb_mcp, thumb_ip, thumb_tip,
index_finger_mcp,  index_finger_pip,  index_finger_dip,  index_finger_tip,
middle_finger_mcp, middle_finger_pip, middle_finger_dip, middle_finger_tip,
ring_finger_mcp,   ring_finger_pip,   ring_finger_dip,   ring_finger_tip,
pinky_mcp,         pinky_pip,         pinky_dip,         pinky_tip
```

So with both hands in frame you get up to **126 messages per frame**. Build your own mapping (frequency, volume, filter cutoff, anything) in your downstream patch by subscribing to whichever addresses you care about.

---

## Mapping coordinates to sound (in your patch)

All values arrive as floats:

- **`x`** — horizontal position, `0.0` (left edge) → `1.0` (right edge)
- **`y`** — vertical position, `0.0` (top edge) → `1.0` (bottom edge)
- **`z`** — depth relative to the wrist; negative = toward the camera, positive = away. Magnitudes are small (roughly `-0.2` to `+0.2`) and not metric.

A few starter ideas you can wire up in Pd/Max/SC:

- **Pitch from horizontal index tip:** read `/right_index_finger_tip_x`, scale `0–1` to your desired frequency range (e.g. `200 + x*1800` for 200–2000 Hz).
- **Volume from vertical:** read `/right_index_finger_tip_y` and invert (`1 - y`) so raising your hand makes it louder.
- **Pinch as gate:** subscribe to both `/right_thumb_tip_x,y` and `/right_index_finger_tip_x,y`, compute distance in your patch, gate the synth when below a threshold.
- **Two-hand control:** use left-hand joints for one synth parameter and right-hand joints for another — they're independent streams.

If you want to do the mapping *in Python instead of in the patch*, the place to add it is inside the `consumer()` function in [doublehand_mp.py](doublehand_mp.py), right where the per-landmark `client.send_message(...)` calls happen — replace or supplement them with whatever derived value you want to send.

---

## Toggling the preview window

At the top of [doublehand_mp.py](doublehand_mp.py):

```python
SHOW_PREVIEW = True   # set to False to hide the camera window
```

Set it to `False` if you want to run headless (e.g. during a live performance).
