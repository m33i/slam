import cv2
import numpy as np
from display import Display
from feat_extractor import FeatExtractor
from mapping import Mapping
import utils
import sys
import os
import time

class SLAM:
    def __init__(self, video_path):
        self.W = 1920 // 2
        self.H = 1080 // 2

        # params
        self.F = float(os.getenv('F', '270')) # focal length
        self.COLORS = os.getenv('COLORS', '0') == '1' # colors
        self.DETECTOR = os.getenv('DETECTOR', 'GFTT') # feature detector type
        self.O3D_OUT = os.getenv('O3D_OUT', '0') == '1' # get 3d visualization outside

        # intrinsic camera matrix
        self.K = np.array(([self.F, 0, self.W//2], [0, self.F, self.H//2], [0, 0, 1]))
        self.Kinv = np.linalg.inv(self.K)  # inverse of K

        self.ui_display = Display(self.W, self.H)
        self.mapping = Mapping(display=self.ui_display, open3d_out=self.O3D_OUT)
        self.feat_extractor = FeatExtractor(K=self.K, detector=self.DETECTOR)

        # open video
        self.vid = video_path
        self.cap = cv2.VideoCapture(self.vid)

        # check if video opened successfully
        if not self.cap.isOpened():
            print("error: Could not open video")
            exit()

        self.last_time = time.time()
        self.frames = 0

        # set up mouse callback
        cv2.setMouseCallback(self.ui_display.window_name, self.mouse_callback)

    # converting 2D points to 3D using camera matrix
    def project_2d_to_3d(self, points_2d, depth=1.0):
        points_2d_homogeneous = np.hstack((points_2d, np.ones((len(points_2d), 1))))
        points_3d = depth * (self.Kinv @ points_2d_homogeneous.T).T
        return points_3d

    def handle_control_change(self, control):
        if control == 'orb':
            new_detector = 'ORB' if self.ui_display.states['orb'] else 'GFTT'
            self.feat_extractor = FeatExtractor(K=self.K, detector=new_detector)
            print(f"\n|| Switched detector to {new_detector}")
        elif control == 'colors':
            self.COLORS = self.ui_display.states['colors']
            print(f"\n|| Colors {'enabled' if self.COLORS else 'disabled'}")
        elif control == 'points_3d':
            print(f"\n|| 3D points {'enabled' if self.ui_display.states['points_3d'] else 'disabled'}")

    def process_frame(self, frame):   
        # resize input frame
        resized_frame = cv2.resize(frame, (self.W, self.H))
        
        # process the frame to extract features and update pose
        matching_display, features_display, features = self.feat_extractor.process_frame(resized_frame)
        pose = self.feat_extractor.get_pose()

        print("pose:\n", pose) # for debugging pose till i am sure it's correct
        
        # update displays
        if features_display is not None:
            self.ui_display.update_display(features_display, 'features')
        
        if matching_display is not None:
            self.ui_display.update_display(matching_display, 'matching')

        # always update trajectory with pose
        if pose is not None:
            if self.ui_display.states['points_3d'] and features is not None and len(features) > 0:
                # convert 2D points to 3D space
                points_3d = self.project_2d_to_3d(features)
                # apply pose transformation to the 3D points
                if points_3d is not None and points_3d.shape[1] == 3:
                    points_3d_homo = np.hstack((points_3d, np.ones((len(points_3d), 1))))
                    points_3d = (pose @ points_3d_homo.T).T[:, :3]
                    
                    # green rgb by default if colors are disabled
                    colors = utils.generate_colors_from_image(features, resized_frame) if self.ui_display.states['colors'] else np.array([[0, 1, 0]] * len(points_3d))
                    self.mapping.update_map(points_3d, colors, pose)
            else:
                self.mapping.update_map([], None, pose)
        return features_display

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            control = self.ui_display.check_click(x, y)
            if control:
                self.handle_control_change(control)

    def run(self):
        try:
            while self.cap.isOpened():
                # read frame, if no frame, end
                ret, frame = self.cap.read()
                if ret:
                    self.process_frame(frame)
                    
                    # measuring frame processing time
                    self.frames += 1
                    if time.time() - self.last_time >= 1.0:
                        self.ui_display.fps = self.frames / (time.time() - self.last_time)
                        self.frames = 0
                        self.last_time = time.time()
                    
                    # exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.mapping.close()

if __name__ == "__main__":
    slam = SLAM(sys.argv[1])
    slam.run()