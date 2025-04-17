import cv2
import numpy as np
from orb_detector import OrbDetector

W = 1920 // 2
H = 1080 // 2

def process_video():
    # open video
    cap = cv2.VideoCapture("video_examples/test_city_bus.mp4")
   
    # check if video opened successfully
    if not cap.isOpened():
        print("err: Could not open video")
        return
   
    # Initialize OrbDetector
    orb_detector = OrbDetector()
   
    while cap.isOpened():
        # read frame, if no frame, end
        ret, frame = cap.read()
       
        if not ret:
            break
       
        frame = cv2.resize(frame, (W, H))
       
        # Process the frame with the OrbDetector
        frame_with_orbs = orb_detector.process_frame(frame)
       
        cv2.imshow('frame', frame_with_orbs)
       
        # press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()