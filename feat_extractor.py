import cv2
import numpy as np

class FeatExtractor:
    def __init__(self, nfeatures=3000, qualityLevel=0.01, minDistance=3):
        # Initialize ORB detector
        self.orb = cv2.ORB_create()
        
        # Parameters for goodFeaturesToTrack
        self.nfeatures = nfeatures
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance

    def process_frame(self, frame):
        if frame is None:
            return None

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect features using goodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(
            gray, 
            self.nfeatures, 
            self.qualityLevel, 
            self.minDistance
        )

        # Draw corners if found
        if corners is not None:
            corners = corners.astype(np.int32)
            for corner in corners:
                x, y = corner[0]
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Green dots

        # 2. Detect and compute ORB features
        keypoints, _ = self.orb.detectAndCompute(gray, None)
        
        # Draw ORB keypoints
        for kp in keypoints:
            x, y = map(int, kp.pt)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue dots

        return frame
