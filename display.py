import cv2
import numpy as np

class Display:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.display_name = 'slam'
        cv2.namedWindow(self.display_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.display_name, 0, 0)
        cv2.resizeWindow(self.display_name, W*2, H*2)
        
        # initialize the displays
        self.displays = {
            'features': np.zeros((H, W, 3), dtype=np.uint8),
            '3d': np.zeros((H, W, 3), dtype=np.uint8),
            'matching': np.zeros((H, W*2, 3), dtype=np.uint8)  # matching has width of 2W
        }
        self.combined_display = np.zeros((H*2, W*2, 3), dtype=np.uint8)

    def update_display(self, frame, display_type):
        if display_type not in self.displays:
            print(f"unknown display type: {display_type}")
            return
            
        if frame is not None:
            # convert to BGR if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # resize based on display type
            target_size = (self.W, self.H)
            if display_type == 'matching':
                target_size = (self.W * 2, self.H)
            
            frame = cv2.resize(frame, target_size)
            self.displays[display_type] = frame
            
        self._update_display()
        return cv2.waitKey(1)
        
    def _update_display(self):
        # combine top row
        top_row = np.hstack([self.displays['features'], self.displays['3d']])
        
        # create combined displays (top row above matching)
        self.combined_display = np.vstack([top_row, self.displays['matching']])
        
        # labels
        cv2.putText(self.combined_display, "Features", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(self.combined_display, "Open3D", (self.W + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
        cv2.imshow(self.display_name, self.combined_display)
    
    def close(self):
        cv2.destroyWindow(self.display_name)