import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
# For webcam input:
cap = cv2.VideoCapture(2)
with mp_pose.Pose(
    min_detection_confidence=0.6,
    enable_segmentation=True,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if not results.pose_landmarks:
      continue
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # Segment human from the image.

    condition =  np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.95
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    masked_image = np.where(condition, image, bg_image)
    _, mask_img = cv2.threshold(cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY),1, 255, cv2.THRESH_BINARY)


    # Draw the pose annotation on the image.
    # image.flags.writeable = True
    # annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        masked_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    print(mask_img)
    cv2.imshow('MediaPipe Pose', cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
