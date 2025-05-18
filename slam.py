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

F = 270  # focal length

ui_display = Display(W, H)

# intrinsic camera matrix
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))
Kinv = np.linalg.inv(K)  # inverse of K

feature_type = os.getenv('FEATURE_TYPE', 'GFTT')
feat_extractor = FeatExtractor(K=K, feature_type=feature_type)

# get 3d visualization outside
VIEW_3D = int(os.getenv('VIEW_3D', '0'))
mapping = Mapping(display=ui_display, show_3d_out=VIEW_3D)


def project_2d_to_3d(points_2d, depth=1.0):
    # converting 2D points to 3D using camera matrix
    points_2d_homogeneous = np.hstack((points_2d, np.ones((len(points_2d), 1))))
    points_3d = depth * (Kinv @ points_2d_homogeneous.T).T
    return points_3d

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

    # update 3D map if points are available
    if features is not None and len(features) > 0:
        # convert 2D points to 3D space
        points_3d = project_2d_to_3d(features)
        
        # apply pose transformation to the 3D points
        if points_3d is not None and points_3d.shape[1] == 3:
            points_3d_homo = np.hstack((points_3d, np.ones((len(points_3d), 1))))
            points_3d_transformed = (pose @ points_3d_homo.T).T
            points_3d = points_3d_transformed[:, :3]
            
            # colors_rgb = utils.generatecolors_from_image(features, frame)
            # mapping.update_map(points_3d, colors_rgb, pose)

            color = np.array([[0, 1, 0]] * len(points_3d)) # green [r,g,b]
            mapping.update_map(points_3d, color, pose)
        else:
            print(f"iInvalid 3D points shape: {points_3d.shape if points_3d is not None else None}")
   
    return features_display

if __name__ == "__main__":
    # open video
    vid = sys.argv[1]
    cap = cv2.VideoCapture(vid)

    # check if video opened successfully
    if not cap.isOpened():
        print("error: Could not open video")
        exit()

    total_time = 0
    frame_count = 0

    try:
        while cap.isOpened():
            # read frame, if no frame, end
            ret, frame = cap.read()
            if ret:
                # measuring frame processing time
                start = time.time()
                process_frame(frame)
                total_time += time.time() - start
                frame_count += 1
                
                # exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mapping.close()
        if frame_count > 0:
            print(f"\raverage time per frame: {total_time/frame_count:.4f} s ({frame_count} frames)")
            # useful to check between orb or gftt etc