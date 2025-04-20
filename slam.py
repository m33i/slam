import cv2
import numpy as np
from display import Display
from feat_extractor import FeatExtractor

W = 1920 // 2
H = 1080 // 2

display = Display(W, H)

def process_frame(img):   
    # Initialize FeatExtractor
    feat_extractor = FeatExtractor()
   
    # Resize the input image
    frame = cv2.resize(img, (W, H))
   
    # Process the frame with the FeatExtractor
    frame_with_feats = feat_extractor.process_frame(frame)
   
    # Display the frame using Display class
    display.show(frame_with_feats)
   
    # Return processed frame
    return frame_with_feats

if __name__ == "__main__":
    # open video
    cap = cv2.VideoCapture("video_examples/test_road.mp4")
    
    # check if video opened successfully
    if not cap.isOpened():
        print("err: Could not open video")
        exit()
    
    while cap.isOpened():
        # read frame, if no frame, end
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
            # Salir si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()