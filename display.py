import cv2
import numpy as np
import os

class Display:
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.panel_width = 210  # width for labels to fit in
        self.window_name = 'Monocular SLAM'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 0, 0)
        cv2.resizeWindow(self.window_name, W*2, H*2)
        
        # initialize the displays
        self.displays = {
            'features': np.zeros((H, W, 3), dtype=np.uint8),
            '3d': np.zeros((H, W, 3), dtype=np.uint8),
            'matching': np.zeros((H, W*2, 3), dtype=np.uint8)  # matching has width of 2W
        }
        self.combined_display = np.zeros((H*2, W*2 + self.panel_width, 3), dtype=np.uint8)
        
        # track states
        self.states = {
            'points_3d': True,
            'colors': os.getenv('COLORS', '0') == '1',
            'orb': os.getenv('DETECTOR', 'GFTT').upper() == 'ORB'
        }
        self.fps = 0

    def draw_checkbox(self, x, y, label, checked):
        # shadow for checkbox
        cv2.rectangle(self.combined_display, (x+3, y-7), (x+27, y+17), (30, 30, 30), -1)
        # checkbox box
        cv2.rectangle(self.combined_display, (x, y-10), (x+20, y+10), (220, 220, 220), 2)
        if checked:
            cv2.line(self.combined_display, (x+5, y), (x+8, y+5), (0, 255, 0), 2)
            cv2.line(self.combined_display, (x+8, y+5), (x+15, y-5), (0, 255, 0), 2)
        
        # draw labels
        cv2.putText(self.combined_display, label, (x+34, y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 2)  # shadow
        cv2.putText(self.combined_display, label, (x+32, y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    def check_click(self, x, y):
        if x < self.panel_width:  # panel width
            y_pos = 38
            for key, label in [('orb', 'ORB'), ('colors', 'Colors'), ('points_3d', '3D Points')]:
                if y_pos-12 <= y <= y_pos+12:
                    self.states[key] = not self.states[key]
                    return key
                y_pos += 48
        return None

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
        
    def _update_display(self):
        # panel background
        self.combined_display[:] = (10, 30, 30) # bg color
        
        # panel checkboxes
        y_pos = 38
        for key, label in [('orb', 'ORB'), ('colors', 'Colors'), ('points_3d', '3D Points')]:
            self.draw_checkbox(18, y_pos, label, self.states[key])
            y_pos += 48
        
        # draw FPS
        cv2.putText(self.combined_display, f"FPS: {self.fps:.1f}", (18, y_pos+18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        
        # main displays
        self.combined_display[:self.H, self.panel_width:self.panel_width+self.W] = self.displays['features']
        self.combined_display[:self.H, self.panel_width+self.W:] = self.displays['3d']
        self.combined_display[self.H:, self.panel_width:] = self.displays['matching']
        
        # display labels
        cv2.putText(self.combined_display, "Features", (self.panel_width+20, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(self.combined_display, "Open3D", (self.panel_width + self.W + 20, 32), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    
        cv2.imshow(self.window_name, self.combined_display)
    
    def close(self):
        cv2.destroyWindow(self.window_name)