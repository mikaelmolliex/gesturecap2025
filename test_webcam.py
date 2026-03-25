from gesturecap2025.video.webcam_input2 import WebcamInput
import cv2

cam = WebcamInput()

while True:
    frame = cam.read()
    if frame is None:
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()