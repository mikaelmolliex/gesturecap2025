"""
preview_webcam.py

Development alternative to preview_flircam.py for machines without
a FLIR Blackfly camera. Uses a standard webcam via WebcamInput.

Use this to:
  - Verify camera positioning before calibration
  - Test the video pipeline without FLIR hardware
  - Debug gesture detection on any laptop

Usage:
    python latency_measurement/preview_webcam.py

Controls:
    Press 'q' to quit.
"""

import sys
import os

# Add project root to path so `video` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from video.webcam_input import WebcamInput


def main():
    print("=" * 50)
    print("GestureCap — Webcam Preview (no FLIR required)")
    print("Press 'q' to quit.")
    print("=" * 50)

    cam = WebcamInput(camera_index=0, width=640, height=480, fps=30)

    last_ts = None
    fps_display = 0.0

    try:
        while True:
            result = cam.read_frame()

            if result is None:
                print("Failed to grab frame.")
                break

            frame, ts, _ = result

            # Calculate FPS from timestamp difference (mirrors preview_flircam.py)
            if last_ts is not None:
                dt = ts - last_ts
                if dt > 0:
                    fps_display = 1.0 / dt
            last_ts = ts

            # Draw the same overlay as preview_flircam.py so behaviour is consistent
            cv2.line(frame, (0, 400), (frame.shape[1], 400), (255, 0, 0), 2)
            cv2.putText(frame, f"TS: {ts:.3f}s", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "[WEBCAM MODE]", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("GestureCap - Webcam Preview", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting preview.")
                break

    except KeyboardInterrupt:
        pass

    finally:
        cam.cleanup()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()