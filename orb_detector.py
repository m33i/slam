import cv2

class OrbDetector:
    def __init__(self, nfeatures=2000):
        # initialize orb detector
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def process_frame(self, frame):
        if frame is None:
            return None

        # detect keypoints and compute descriptors
        keypoints = self.orb.detect(frame, None)

        # draw keypoints on the frame
        frame_with_kp = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        return frame_with_kp
