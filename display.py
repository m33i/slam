import cv2

class Display:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.window_name = 'slam'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # opencv pos to the left
        cv2.moveWindow(self.window_name, 0, 0)
        cv2.resizeWindow(self.window_name, W, H)

    def show(self, frame):
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)