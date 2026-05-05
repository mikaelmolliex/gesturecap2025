# Quickstart

This guide gets you from a blank machine to running hand-gesture-to-OSC mapping in a few minutes.

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

## What the hands do

| Hand | Gesture | OSC message sent |
|------|---------|-----------------|
| **Left** | Pinch index + thumb together | `/trigger 1` (one shot per pinch) |
| **Right** | Pinch index + thumb, then move hand | `/frequency <f>` and `/volume <v>` continuously |

---

## Tweaking the musical mapping (right hand)

The two lines that turn hand position into sound are in [doublehand_mp.py](doublehand_mp.py) inside the `consumer()` function, in the `elif label == "right":` block:

```python
freq   = 100000 / ((index_pos.x ** 2) * 1000 + 100)
volume = index_pos.y
```

- **`index_pos.x`** is the horizontal position of your index fingertip — `0.0` = left edge of frame, `1.0` = right edge.
- **`index_pos.y`** is the vertical position — `0.0` = top of frame, `1.0` = bottom.

**`freq`** — the formula maps a wide horizontal sweep to a frequency range roughly 1 Hz – 1000 Hz (higher toward the left, lower toward the right). To change the range, adjust the constants:

```python
# Example: linear map, 200 Hz on the left → 2000 Hz on the right
freq = 200 + index_pos.x * 1800
```

**`volume`** — raw `y` means the top of frame is quietest (0.0) and the bottom is loudest (1.0). Flip it if you prefer top = loud:

```python
volume = 1.0 - index_pos.y
```

The pinch threshold that activates the right hand is the `dist < 0.1` check just above those lines. Increase `0.1` to trigger with a looser pinch, decrease it to require a tighter one.

---

## Toggling the preview window

At the top of [doublehand_mp.py](doublehand_mp.py):

```python
SHOW_PREVIEW = True   # set to False to hide the camera window
```

Set it to `False` if you want to run headless (e.g. during a live performance).
