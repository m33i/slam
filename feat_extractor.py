import cv2
import numpy as np

class FeatExtractor:
    def __init__(self, nfeatures=3000, qualityLevel=0.01, minDistance=3):
        # init ORB detector
        self.orb = cv2.ORB_create()
        
        # params for goodFeaturesToTrack
        self.nfeatures = nfeatures
        self.qualityLevel = qualityLevel
        self.minDistance = minDistance

    def process_frame(self, frame):
        if frame is None:
            return None, None

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect features using goodFeaturesToTrack
        corners = cv2.goodFeaturesToTrack(
            gray, 
            self.nfeatures, 
            self.qualityLevel, 
            self.minDistance
        )

        frame_with_features = frame.copy()
        points = []

        # draw corners if found
        if corners is not None:
            corners = corners.astype(np.int32)
            points = [corner[0] for corner in corners]  # Extract points
            for corner in corners:
                x, y = corner[0]
                cv2.circle(frame_with_features, (x, y), 3, (0, 255, 0), -1)

        # detect and compute ORB features
        keypoints, _ = self.orb.detectAndCompute(gray, None)
        
        # add ORB keypoints to points list
        for kp in keypoints:
            x, y = map(int, kp.pt)
            points.append([x, y])
            cv2.circle(frame_with_features, (x, y), 2, (255, 0, 0), -1)

        return frame_with_features, np.array(points)
