import open3d as o3d
import numpy as np

def viewc_settings(vis):
    view_control = vis.get_view_control()
    view_control.set_front([0, -1, 0])
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(1.2)

def vis_settings(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2

# show keyframe as a wireframe frustums square
def show_keyframe_square(vis, trajectory, K, start_index=0):
    if start_index >= len(trajectory):
        return
        
    points = np.asarray(trajectory)[start_index:]

    # calculate template using camera intrinsics
    z_near, z_far = 0.1, 0.2
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    template = np.array([
        [0, 0, 0],  # camera center
        # near rectangle
        [z_near * (-cx)/fx, z_near * (-cy)/fy, z_near],
        [z_near * (640-cx)/fx, z_near * (-cy)/fy, z_near],
        [z_near * (640-cx)/fx, z_near * (480-cy)/fy, z_near],
        [z_near * (-cx)/fx, z_near * (480-cy)/fy, z_near],
        # far rectangle (scaled version of near)
        [z_far * (-cx)/fx, z_far * (-cy)/fy, z_far],
        [z_far * (640-cx)/fx, z_far * (-cy)/fy, z_far],
        [z_far * (640-cx)/fx, z_far * (480-cy)/fy, z_far],
        [z_far * (-cx)/fx, z_far * (480-cy)/fy, z_far]
    ])

    # define edges/corners
    _edges = [
        [0,1], [0,2], [0,3], [0,4],  # lines from center
        [1,2], [2,3], [3,4], [4,1],  # near rectangle
        [5,6], [6,7], [7,8], [8,5],  # far rectangle
        [1,5], [2,6], [3,7], [4,8]   # connecting lines
    ]
    
    vertices = np.tile(template, (len(points), 1)) + np.repeat(points, len(template), axis=0)
    edges = np.array(_edges) + np.arange(0, len(points) * len(template), len(template))[:, None, None]
    edges = edges.reshape(-1, 2)
    
    frustums = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edges)
    )
    frustums.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in range(len(edges))])
    vis.add_geometry(frustums)

# debugging stuff
def camera_position_callback(vis):
    view_control = vis.get_view_control()
    camera_position = view_control.convert_to_pinhole_camera_parameters().extrinsic[:3, 3]
    print(f"\rcamera position: ({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})", end="")
    return False

def generate_colors_from_image(features, frame):
    #generate colors based on pixel values
    valid_y = np.clip(features[:, 1], 0, frame.shape[0]-1).astype(int)
    valid_x = np.clip(features[:, 0], 0, frame.shape[1]-1).astype(int)
    colors_bgr = frame[valid_y, valid_x]
    colors_rgb = colors_bgr[..., ::-1] / 255.0  # bgr to rgb normalized
    return colors_rgb

def get_mask_and_line(keypoints, height, width, f_mask=1.0, sky_auto=True):
    if abs(f_mask) < 1.0:
        mask_portion = abs(f_mask)
    elif sky_auto and any(kp.pt[1] < height * 0.2 for kp in keypoints):
        mask_portion = 0.6
        f_mask = 0.6
    else:
        mask_portion = 1.0

    mask = None
    if abs(f_mask) < 1.0:
        mask = np.zeros((height, width), dtype=np.uint8)
        if f_mask > 0:
            # upper part
            start_row = int(height * (1 - mask_portion))
            mask[start_row:, :] = 255
        else:
            # lower part
            end_row = int(height * mask_portion)
            mask[:end_row, :] = 255
    elif mask_portion < 1.0:
        # auto
        mask = np.zeros((height, width), dtype=np.uint8)
        start_row = int(height * (1 - mask_portion))
        mask[start_row:, :] = 255
    return mask, mask_portion