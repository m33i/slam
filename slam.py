import cv2
import numpy as np
from display import Display
from feat_extractor import FeatExtractor
from mapping import Mapping

W = 1920 // 2
H = 1080 // 2

display = Display(W, H)
feat_extractor = FeatExtractor() 
mapping = Mapping()

F = 270  # focal length

K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))  # intrinsic camera matrix
# inverse of intrinsic camera matrix
Kinv = np.linalg.inv(K)

def project_2d_to_3d(points_2d, depth=1.0):
    # converting 2D points to 3D using camera matrix
    points_2d_homogeneous = np.hstack((points_2d, np.ones((len(points_2d), 1))))
    points_3d = depth * (Kinv @ points_2d_homogeneous.T).T
    return points_3d

def process_frame(img):   
    # resize input
    frame = cv2.resize(img, (W, H))
   
    # process the frame to extract features
    frame_with_feats, points = feat_extractor.process_frame(frame)
    
    # update 3D map if points are available
    if points is not None and len(points) > 0:
        # 2D points to 3D space
        points_3d = project_2d_to_3d(points)
        
        # Verify points_3d is valid
        if points_3d is not None and points_3d.shape[1] == 3:
            # generated colors based on pixel values
            try:
                valid_y = np.clip(points[:, 1], 0, frame.shape[0]-1).astype(int)
                valid_x = np.clip(points[:, 0], 0, frame.shape[1]-1).astype(int)

                colors = frame[valid_y, valid_x] / 255.0

                mapping.update_map(points_3d, colors)
            except IndexError:
                print("[Warning] Could not generate colors for some points - skipping frame")
        else:
            print(f"[Warning] Invalid 3D points shape: {points_3d.shape if points_3d is not None else None}")
   
    # display the frame
    if frame_with_feats is not None:
        display.show(frame_with_feats)
   
    return frame_with_feats

if __name__ == "__main__":
    # open video
    cap = cv2.VideoCapture("video_examples/test_road.mp4")
    
    # check if video opened successfully
    if not cap.isOpened():
        print("err: Could not open video")
        exit()
    
    try:
        while cap.isOpened():
            # read frame, if no frame, end
            ret, frame = cap.read()
            if ret == True:
                process_frame(frame)
                # exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        mapping.close()