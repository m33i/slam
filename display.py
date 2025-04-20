import cv2

class Display:
    def __init__(self, W, H):
        self.W = W
        self.H = H

    def show(self, frame, window_name='frame'):
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1)