import cv2
import numpy as np
import logging
import time
from video.video_input import VideoInput

logger = logging.getLogger(__name__)


class WebcamInput(VideoInput):
    """
    Webcam-based video input using OpenCV (cv2.VideoCapture).

    This is a drop-in replacement for Flircam for development and testing
    on machines without a FLIR Blackfly camera. It implements the same
    VideoInput ABC so the rest of the pipeline works unchanged.

    Parameters
    ----------
    camera_index : int
        Index of the webcam to open. 0 is usually the built-in webcam.
    width : int
        Requested frame width in pixels.
    height : int
        Requested frame height in pixels.
    fps : int
        Requested frames per second.

    Usage
    -----
    cam = WebcamInput(camera_index=0, width=640, height=480, fps=30)
    frame, ts, timing = cam.read_frame()
    cam.cleanup()
    """

    def __init__(self, camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self._cap = None
        # NOTE: super().__init__() must be called at the END
        # because it calls configure() which needs self._cap ready
        super().__init__()

    def configure(self) -> None:
        """
        Opens the webcam and applies resolution + FPS settings.
        Called automatically by VideoInput.__init__() via super().__init__().
        """
        self._cap = cv2.VideoCapture(self.camera_index)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam at index {self.camera_index}. "
                "Check that a webcam is connected and not used by another app."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"[WebcamInput] Opened webcam {self.camera_index}: "
            f"{actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
        )
        print(
            f"[WebcamInput] Opened webcam {self.camera_index}: "
            f"{actual_w}x{actual_h} @ {actual_fps:.1f} FPS"
        )

    def read_frame(self):
        """
        Captures one frame from the webcam.

        Returns the same 3-tuple as Flircam.read_frame() so the rest of
        the pipeline needs zero changes when swapping camera backends.

        Returns
        -------
        frame : np.ndarray
            BGR image as a writable uint8 array of shape (H, W, 3).
        ts : float
            Timestamp in seconds (from time.perf_counter at capture moment).
        timing : tuple of float
            (t_frameacq, t_getts, t_frameconv) — time taken for each internal
            step, in seconds. t_getts and t_frameconv are 0.0 for webcam
            since there is no chunk metadata or color conversion step.
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(
                "Webcam is not open. Make sure configure() ran successfully."
            )

        t0 = time.perf_counter()
        ret, frame = self._cap.read()
        t1 = time.perf_counter()

        if not ret or frame is None:
            logger.warning("[WebcamInput] Failed to read frame.")
            return None

        ts = time.perf_counter()   # wall-clock timestamp in seconds
        t2 = time.perf_counter()

        # Make frame explicitly writable (mirrors Flircam behaviour)
        frame = np.array(frame, dtype=np.uint8)

        t3 = time.perf_counter()

        t_frameacq = t1 - t0      # time to grab frame from hardware
        t_getts = t2 - t1         # time to get timestamp (trivial for webcam)
        t_frameconv = t3 - t2     # time to ensure writeable array

        return frame, ts, (t_frameacq, t_getts, t_frameconv)

    def cleanup(self) -> None:
        """
        Releases the webcam. Mirrors Flircam.cleanup() behaviour.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("[WebcamInput] Webcam released.")
            print("[WebcamInput] Webcam released.")