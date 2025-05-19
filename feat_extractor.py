import cv2
import numpy as np
import os
import utils

class FeatExtractor:
    # experimental to avoid sky features and so on
    F_MASK = float(os.getenv('F_MASK', '1.0'))
    SKY_AUTO = int(os.getenv('SKY_AUTO', '0'))

    def __init__(self, K=None, detector='GFTT'):
        self.detector = detector.upper()
        if self.detector == 'ORB':
            self.orb = cv2.ORB_create(
                nfeatures=3000,
                scaleFactor=1.2,
                nlevels=8
            )
            self.gftt = None
            self.brief = None
        else:  # GFTT + BRIEF
            self.gftt = cv2.GFTTDetector_create(
                maxCorners=3000,
                qualityLevel=0.02,
                minDistance=10,
                blockSize=7,
                #useHarrisDetector=True,  # Harris not really doing much in our case
                #k=0.04
            )
            self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32,
                use_orientation=True # use orientation for better matching
            )
            self.orb = None

        self.K = K # intrinsic camera matrix
        self.good_matches_count = 0

        # setup for FLANN based matcher 
        FLANN_INDEX_LSH = 6
        index_params = dict( # basically high values are better for matching (but slower) 
            algorithm=FLANN_INDEX_LSH, 
            table_number=16, 
            key_size=24, 
            multi_probe_level=2)
        search_params = dict(checks=100)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.prev_img = None
        self.prev_kps = None
        self.prev_descriptors = None
        self.curr_pose = np.eye(4)  # starting pose (identity matrix)
        
        # simple motion model for prediction
        self.last_R = np.eye(3)
        self.last_t = np.zeros((3, 1))
        self.velocity = np.zeros((3, 1))
        self.scale_factor = 0.05  # scale for translation

    def process_frame(self, frame):
        if frame is None:
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_with_features = frame.copy()
        matching_visualization = None
        
        if self.detector == 'ORB':
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        else:
            # detect GFTT keypoints
            pts = self.gftt.detect(gray, None)
            keypoints = pts if pts is not None else []
            if len(keypoints) > 0:
                keypoints, descriptors = self.brief.compute(gray, keypoints)
            else:
                descriptors = None

            mask = utils.get_mask_and_line(keypoints, gray.shape[0], gray.shape[1], self.F_MASK, self.SKY_AUTO)

            if mask is not None:
                pts = self.gftt.detect(gray, mask)
                keypoints = pts if pts is not None else []
                if len(keypoints) > 0:
                    keypoints, descriptors = self.brief.compute(gray, keypoints)
                else:
                    descriptors = None
        
        # draw features on visualization frame
        for kp in keypoints:
            x, y = map(int, kp.pt)
            cv2.circle(frame_with_features, (x, y), 2, (0, 255, 0), -1)
        
        # get point coordinates for 3D mapping
        points = np.array([kp.pt for kp in keypoints]) if len(keypoints) > 0 else np.empty((0,2))
        
        # storing first frame
        if self.prev_img is None or self.prev_descriptors is None or descriptors is None:
            self.prev_img = gray.copy()
            self.prev_kps = keypoints
            self.prev_descriptors = descriptors
            return None, frame_with_features, points
        
        # match descriptors between frames
        matches = self.match_descriptors(self.prev_descriptors, descriptors)
        
        # draw feature matching visualization
        matching_visualization = self.draw_matches(gray, keypoints, matches)
        
        # if we have enough matches we can estimate motion
        if len(matches) >= 8:
            # get matched point coordinates
            prev_pts = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])
            
            # estimate camera motion
            success, R, t = self.estimate_transform(prev_pts, curr_pts)
            
            if success:
                # scale translation to keep motion reasonable
                t = t * self.scale_factor
                
                # update velocity with exponential smoothing
                self.velocity = 0.8 * self.velocity + 0.2 * t
                
                # store current transformation
                self.last_R = R
                self.last_t = t
                
                # transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.ravel()
                
                # update pose (camera moves in opposite direction)
                self.curr_pose = np.dot(self.curr_pose, np.linalg.inv(T))
                #print(f"motion estimated: translation = {np.linalg.norm(t):.3f}")
            else:
                # using predicted motion based on previous velocity
                T = np.eye(4)
                T[:3, 3] = self.velocity.ravel()
                self.curr_pose = np.dot(self.curr_pose, np.linalg.inv(T))
                #print(f"using motion model: translation = {np.linalg.norm(self.velocity):.3f}")
        
        # update previous frame data
        self.prev_img = gray.copy()
        self.prev_kps = keypoints
        self.prev_descriptors = descriptors
        
        return matching_visualization, frame_with_features, points

    def match_descriptors(self, des1, des2):
        try:
            # making sure descriptors are valid
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                return []
            
            # match using FLANN
            matches = self.flann.knnMatch(des1, des2, k=2)
            
            # Lowe's ratio test 
            # keep only good matches
            good_matches = []
            for match_group in matches:
                if len(match_group) == 2:
                    m, n = match_group
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            self.good_matches_count = len(good_matches)
            return good_matches
        except Exception as e:
            print(f"error in matching: {e}")
            return []

    def estimate_transform(self, src_pts, dst_pts):
        try:
            if len(src_pts) < 8 or len(dst_pts) < 8:
                return False, None, None

            # filter correspondences using RANSAC (essential matrix)
            E, mask = cv2.findEssentialMat(
                src_pts, dst_pts, self.K, 
                method=cv2.RANSAC, 
                prob=0.999, 
                threshold=1.0
            )
            
            if E is None or mask is None:
                return False, None, None
            
            # inliers
            inliers_count = np.sum(mask)
            if inliers_count < 6:  # at least 6 good points
                return False, None, None
            
            # recover pose from essential matrix
            _, R, t, new_mask = cv2.recoverPose(E, src_pts, dst_pts, self.K, mask=mask)
            
            # check if rotation is reasonable (not too large)
            angle = np.arccos((np.trace(R) - 1) / 2)
            if angle > 0.3:  # max 0.3 radians (about 17 degrees) rotation between frames
                print(f"rotation too large: {angle:.2f} rad")
                return False, None, None
            
            return True, R, t
        except Exception as e:
            print(f"error estimating transform: {e}")
            return False, None, None

    def draw_matches(self, current_frame, current_kps, matches):
        try:
            if self.prev_kps is None or len(matches) == 0:
                return None
            
            # sort matches, take top 50
            matches = sorted(matches, key=lambda x: x.distance)
            matches = matches[:50] if len(matches) > 50 else matches
            
            # draw matches
            match_img = cv2.drawMatches(
                self.prev_img, self.prev_kps,
                current_frame, current_kps,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            cv2.putText(match_img, f"matches: {self.good_matches_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return match_img
        except Exception as e:
            print(f"error drawing matches: {e}")

    def get_pose(self):
        return self.curr_pose # debugger