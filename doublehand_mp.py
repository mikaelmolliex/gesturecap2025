"""
doublehand_mp.py

Multiprocess doublehand OSC mapping.

Producer: captures frames from webcam into shared double buffer.
Consumer: detects both hands, sends OSC messages:
  - Left hand pinch  (index-thumb dist < 0.08) → /trigger 1
  - Right hand pinch (index-thumb dist < 0.1)  → /frequency <f>, /volume <v>
"""

import math
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

    left_state  = 0   # hysteresis state for left-hand trigger
    left_count  = 0

    print("CONSUMER: starting doublehand detection — press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            read_idx = cur_idx.value
            frame    = buf0.copy() if read_idx == 0 else buf1.copy()

            hands = detector.detect_hand_pose(frame)

            left_tapped = False

            if hands:
                for hand in hands:
                    label = hand.get("label", "").lower()
                    if not label:
                        continue

                    landmarks = hand["landmarks"].landmark

                    if label == "left":
                        index_pos = landmarks[8]
                        thumb_pos = landmarks[4]
                        dist = math.dist(
                            [index_pos.x, index_pos.y],
                            [thumb_pos.x, thumb_pos.y],
                        )

                        if dist >= 0.08 and left_state == 1:
                            left_state = 0
                        elif dist < 0.08 and left_state == 0:
                            left_state  = 1
                            left_tapped = True
                            left_count += 1
                            print(f"Left hand tap #{left_count}")

                    elif label == "right":
                        index_pos = landmarks[8]
                        thumb_pos = landmarks[4]
                        dist = math.dist(
                            [index_pos.x, index_pos.y],
                            [thumb_pos.x, thumb_pos.y],
                        )

                        if dist < 0.1:
                            freq   = 100000 / ((index_pos.x ** 2) * 1000 + 100)
                            volume = index_pos.y
                            client.send_message("/frequency", freq)
                            client.send_message("/volume",    volume)

            if left_tapped:
                client.send_message("/trigger", 1)

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
