"""
doublehand_mp.py

Multiprocess doublehand OSC mapping.

Producer: captures frames from webcam into shared double buffer.
Consumer: detects both hands and streams every landmark over OSC.

For each detected hand, 63 OSC messages are sent per frame — one per
axis per MediaPipe joint, each carrying a single float:

    /left_wrist_x            <float>
    /left_wrist_y            <float>
    /left_wrist_z            <float>
    /left_thumb_cmc_x        <float>
    ...
    /right_pinky_tip_z       <float>

x, y are normalized to [0, 1] (image-space); z is depth relative to the
wrist (smaller = closer to camera).
"""

import time
import numpy as np
import cv2

from multiprocessing import shared_memory, Event, Process, Value
import multiprocessing as mp

from pythonosc import udp_client

from video.webcam_input import WebcamInput
from utils.hand_pose_detector import HandPoseDetector


# ── Frame buffer config ───────────────────────────────────────────────────────
FRAME_SHAPE = (480, 640, 3)
FRAME_DTYPE = np.uint8

OSC_IP   = "127.0.0.1"
OSC_PORT = 11111

SHOW_PREVIEW = True   # set to False to disable the landmark preview window


# ── MediaPipe hand joint names (index → name) ────────────────────────────────
# Lowercase MediaPipe HandLandmark enum names — used to build OSC addresses
# like /left_thumb_tip, /right_index_finger_mcp, etc.
JOINT_NAMES = [
    "wrist",
    "thumb_cmc",        "thumb_mcp",        "thumb_ip",         "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp","middle_finger_pip","middle_finger_dip","middle_finger_tip",
    "ring_finger_mcp",  "ring_finger_pip",  "ring_finger_dip",  "ring_finger_tip",
    "pinky_mcp",        "pinky_pip",        "pinky_dip",        "pinky_tip",
]


# ── Drawing helpers ───────────────────────────────────────────────────────────
# MediaPipe hand connections (pairs of landmark indices)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]

_COLORS = {
    "left":  (  0, 200,   0),   # green
    "right": (200,   0, 200),   # purple
}


def draw_hand(frame, landmarks, label):
    h, w = frame.shape[:2]
    color = _COLORS.get(label, (255, 255, 255))

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        r = 5 if i in (4, 8) else 3   # bigger dot on thumb/index tips
        cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)

    # Label near wrist
    cv2.putText(frame, label.upper(), pts[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


# ── Producer ──────────────────────────────────────────────────────────────────
def producer(shm_name0, shm_name1, cur_idx, stop_event, ts_value,
             t_read_total_v, t_frameacq_v, t_getts_v, t_frameconv_v):

    cam  = WebcamInput(width=FRAME_SHAPE[1], height=FRAME_SHAPE[0])
    shm0 = shared_memory.SharedMemory(name=shm_name0)
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    buf0 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm0.buf)
    buf1 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm1.buf)

    try:
        while not stop_event.is_set():
            t_start = time.perf_counter()
            result  = cam.read_frame()
            t_end   = time.perf_counter()

            if result is None:
                stop_event.set()
                break

            frame, _ts, (t_frameacq, t_getts, t_frameconv) = result

            # Resize if webcam returned unexpected dimensions
            if frame.shape != tuple(FRAME_SHAPE):
                frame = cv2.resize(frame, (FRAME_SHAPE[1], FRAME_SHAPE[0]))

            write_idx = 1 - cur_idx.value
            if write_idx == 0:
                np.copyto(buf0, frame)
            else:
                np.copyto(buf1, frame)

            t_read_total_v.value = t_end - t_start
            t_frameacq_v.value   = t_frameacq
            t_getts_v.value      = t_getts
            t_frameconv_v.value  = t_frameconv
            ts_value.value       = t_end
            cur_idx.value        = write_idx

    except KeyboardInterrupt:
        print("PRODUCER: KeyboardInterrupt")
    finally:
        cam.cleanup()
        shm0.close()
        shm1.close()
        print("PRODUCER EXITS GRACEFULLY")


# ── Consumer ──────────────────────────────────────────────────────────────────
def consumer(shm_name0, shm_name1, cur_idx, stop_event, ts_value,
             t_read_total_v, t_frameacq_v, t_getts_v, t_frameconv_v):

    client   = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    detector = HandPoseDetector(n_hands=2)

    shm0 = shared_memory.SharedMemory(name=shm_name0)
    shm1 = shared_memory.SharedMemory(name=shm_name1)
    buf0 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm0.buf)
    buf1 = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm1.buf)

    time.sleep(0.5)  # warm-up

    print("CONSUMER: streaming hand landmarks over OSC — press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            read_idx = cur_idx.value
            frame    = buf0.copy() if read_idx == 0 else buf1.copy()

            hands = detector.detect_hand_pose(frame)

            if hands:
                for hand in hands:
                    label = hand.get("label", "").lower()
                    if label not in ("left", "right"):
                        continue

                    landmarks = hand["landmarks"].landmark
                    for i, name in enumerate(JOINT_NAMES):
                        lm = landmarks[i]
                        client.send_message(f"/{label}_{name}_x", float(lm.x))
                        client.send_message(f"/{label}_{name}_y", float(lm.y))
                        client.send_message(f"/{label}_{name}_z", float(lm.z))

            if SHOW_PREVIEW:
                preview = frame.copy()
                if hands:
                    for hand in hands:
                        label = hand.get("label", "").lower()
                        if label:
                            draw_hand(preview, hand["landmarks"].landmark, label)

                cv2.imshow("doublehand_mp", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break

    except KeyboardInterrupt:
        print("CONSUMER: KeyboardInterrupt")
    finally:
        stop_event.set()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        shm0.close()
        shm1.close()
        print("CONSUMER EXITS GRACEFULLY")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)

    size = int(np.prod(FRAME_SHAPE) * np.dtype(FRAME_DTYPE).itemsize)
    shm0 = shared_memory.SharedMemory(create=True, size=size)
    shm1 = shared_memory.SharedMemory(create=True, size=size)

    cur_idx      = Value("i", 0)
    ts           = Value("d", 0.0)
    stop_event   = Event()
    t_read_total = Value("d", 0.0)
    t_frameacq   = Value("d", 0.0)
    t_getts      = Value("d", 0.0)
    t_frameconv  = Value("d", 0.0)

    shared_args = (shm0.name, shm1.name,
                   cur_idx, stop_event, ts,
                   t_read_total, t_frameacq, t_getts, t_frameconv)

    p1 = Process(target=producer, args=shared_args)
    p2 = Process(target=consumer, args=shared_args)

    p1.start()
    p2.start()

    try:
        while p1.is_alive():
            p1.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        p1.join(timeout=2.0)
        p2.join(timeout=2.0)

        for shm in (shm0, shm1):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

        print("MAIN EXIT")
