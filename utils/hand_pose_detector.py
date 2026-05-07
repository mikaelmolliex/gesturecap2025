import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from typing import List

from mediapipe.tasks.python.components.containers import landmark as landmark_module

from mediapipe.framework.formats import landmark_pb2

def convert_to_landmark_list(normalized_landmarks: List[landmark_module.NormalizedLandmark]) -> landmark_pb2.NormalizedLandmarkList:
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for landmark in normalized_landmarks:
        new_landmark = landmark_list.landmark.add()
        new_landmark.x = landmark.x
        new_landmark.y = landmark.y
        new_landmark.z = landmark.z
    return landmark_list


class TempHandLandmarks:
    def __init__(self, landmark_list):
        self.landmark = landmark_list

class HandPoseDetector:
    def __init__(self, n_hands=1, device: str = 'cpu'):
        """
        Initializes the HandLandmarker.


        Parameters:
        ---
        n_hands: int, default=2
            The maximum number of hands to detect in each frame

        device: str, default = 'cpu'
            The device to run the model on. Choose between 'cpu' and 'gpu'.
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        base_options = python.BaseOptions(
            model_asset_path='models/hand_landmarker.task',  # You'll need to download this model
            delegate=python.BaseOptions.Delegate.GPU if device == 'gpu' else python.BaseOptions.Delegate.CPU
            # delegate=python.BaseOptions.Delegate.GPU
            # delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=n_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)

    def detect_hand_pose(self, image): # image is pass by reference, any operations done to the frame inside this method will be reflected in method call origin.
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image using MediaPipe Hands
        # results = self.hands.process(image_rgb)
        results = self.hands.detect(mp_image)

        output = []
        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                # Get the hand label ("Left" or "Right")
                hand = {"label": None, "landmarks": None}
                label = handedness[0].display_name
                # Draw landmarks on the image
                lms_list = list(convert_to_landmark_list(hand_landmarks).landmark)
                
                # for landmark in lms_list:
                #     x = int(landmark.x * image.shape[1])
                #     y = int(landmark.y * image.shape[0])
                #     z = int(landmark.z * image.shape[1])
                #     cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                # Attach label to the hand landmarks object
                hand["label"] = label
                hand["landmarks"] = TempHandLandmarks(lms_list)
                output.append(hand)
        return output

def main():
    # Initialize the hand pose detector
    hand_pose_detector = HandPoseDetector()

    # Open a video capture stream (you can replace this with your own image or video input)
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Break the loop if the video stream ends
        if not ret:
            break

        # Detect hand pose in the frame
        frame_with_landmarks = hand_pose_detector.detect_hand_pose(frame)

        # Display the frame with hand landmarks
        cv2.imshow("Hand Pose Detector", frame_with_landmarks)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
