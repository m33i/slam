import cv2
import numpy as np
from display import Display
from feat_extractor import FeatExtractor
from mapping import Mapping
import utils
import sys
import os
import time

W = 1920 // 2
H = 1080 // 2

F = float(os.getenv('F', '270')) # focal length
COLORS = os.getenv('COLORS', '0') == '1'

ui_display = Display(W, H)

# intrinsic camera matrix
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))
Kinv = np.linalg.inv(K)  # inverse of K

# get 3d visualization outside
O3D_OUT = os.getenv('O3D_OUT', '0') == '1'
mapping = Mapping(display=ui_display, open3d_out=O3D_OUT)

# feature detector type
DETECTOR = os.getenv('DETECTOR', 'GFTT')
feat_extractor = FeatExtractor(K=K, detector=DETECTOR)

def project_2d_to_3d(points_2d, depth=1.0):
    # converting 2D points to 3D using camera matrix
    points_2d_homogeneous = np.hstack((points_2d, np.ones((len(points_2d), 1))))
    points_3d = depth * (Kinv @ points_2d_homogeneous.T).T
    return points_3d

def handle_control_change(control):
    global feat_extractor, mapping, COLORS
    if control == 'orb':
        new_detector = 'ORB' if ui_display.states['orb'] else 'GFTT'
        feat_extractor = FeatExtractor(K=K, detector=new_detector)
        print(f"\n|| Switched detector to {new_detector}")
    elif control == 'colors':
        COLORS = ui_display.states['colors']
        print(f"\n|| Colors {'enabled' if COLORS else 'disabled'}")
    elif control == 'points_3d':
        print(f"\n|| 3D points {'enabled' if ui_display.states['points_3d'] else 'disabled'}")

def process_frame(frame):   
    # resize input frame
    resized_frame = cv2.resize(frame, (W, H))
    
    # process the frame to extract features and update pose
    matching_display, features_display, features = feat_extractor.process_frame(resized_frame)
    pose = feat_extractor.get_pose()

    print("pose:\n", pose) # for debugging pose till i am sure it's correct
    
    # update displays
    if features_display is not None:
        ui_display.update_display(features_display, 'features')
    
    if matching_display is not None:
        ui_display.update_display(matching_display, 'matching')

    # always update trajectory with pose
    if pose is not None:
        if ui_display.states['points_3d'] and features is not None and len(features) > 0:
            # convert 2D points to 3D space
            points_3d = project_2d_to_3d(features)
            # apply pose transformation to the 3D points
            if points_3d is not None and points_3d.shape[1] == 3:
                points_3d_homo = np.hstack((points_3d, np.ones((len(points_3d), 1))))
                points_3d = (pose @ points_3d_homo.T).T[:, :3]
                
                # green rgb by default if colors are disabled
                colors = utils.generate_colors_from_image(features, resized_frame) if ui_display.states['colors'] else np.array([[0, 1, 0]] * len(points_3d))
                mapping.update_map(points_3d, colors, pose)
        else:
            mapping.update_map([], None, pose)
    return features_display

if __name__ == "__main__":
    # open video
    vid = sys.argv[1]
    cap = cv2.VideoCapture(vid)

    # check if video opened successfully
    if not cap.isOpened():
        print("error: Could not open video")
        exit()

    last_time = time.time()
    frames = 0

    # set up mouse callback
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            control = ui_display.check_click(x, y)
            if control:
                handle_control_change(control)

    cv2.setMouseCallback(ui_display.window_name, mouse_callback)

    try:
        while cap.isOpened():
            # read frame, if no frame, end
            ret, frame = cap.read()
            if ret:
                process_frame(frame)
                
                # measuring frame processing time
                frames += 1
                if time.time() - last_time >= 1.0:
                    ui_display.fps = frames / (time.time() - last_time)
                    frames = 0
                    last_time = time.time()
                
                # exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mapping.close()