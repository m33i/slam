import open3d as o3d
import numpy as np

def _vis_settings(vis):
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0  # not too big

class Mapping:
    def __init__(self, display=None, show_3d_out=False):
        self.display = display
        self.show_3d_out = show_3d_out

        self.vis = o3d.visualization.Visualizer()
        # 3d window pos is set to the right
        self.vis.create_window(window_name="3D Map", width=960, height=540, visible=self.show_3d_out)
        
        # initialize an empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # background color and point size
        _vis_settings(self.vis)

        # flag
        self.first_update = True
        
        # store trajectory
        self.trajectory = []
        self.trajectory_line = None
        
        # initialize trajectory with origin
        self.trajectory.append([0, 0, 0])

    def update_map(self, points, colors=None, *args):
        if len(points) == 0:
            return

        # convert points to Open3D format point Cloud
        points = np.asarray(points)
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(points)
        
        # if colors is not None:
        #     current_pcd.colors = o3d.utility.Vector3dVector(colors)
        # else:
        #     current_pcd.paint_uniform_color([1, 1, 1])

        current_pcd.paint_uniform_color([0, 1, 0]) # green [r,g,b]

        # if first update, set the current point cloud as the initial point cloud
        if self.first_update:
            self.pcd.points = current_pcd.points
            self.pcd.colors = current_pcd.colors
            self.first_update = False
        else:
            # merge with existing point cloud
            points = np.vstack((np.asarray(self.pcd.points), np.asarray(current_pcd.points)))
            colors = np.vstack((np.asarray(self.pcd.colors), np.asarray(current_pcd.colors)))
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # update trajectory and current position
        if len(args) > 0:
            pose = args[0]
            current_position = pose[:3, 3]
            self.trajectory.append(current_position)
            self.draw_trajectory()

            # camera view (right now looking at the current position and facing the negative z-axis)
            # TODO: set the camera to look at a different angle (maybe looking from the top like a drone)
            view_control = self.vis.get_view_control()
            view_control.set_lookat(current_position)
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            #view_control.set_zoom(0.1)  # we can use this to look like we are driving inside the street

        # update geometry
        self.vis.update_geometry(self.pcd)

        # update visualization
        self.vis.poll_events()
        self.vis.update_renderer()
        
        # capture visualization for display
        if self.display is not None and not self.show_3d_out:
            img = self.vis.capture_screen_float_buffer(do_render=True)
            img_np = (np.asarray(img) * 255).astype(np.uint8)
            img_bgr = img_np[..., ::-1]
            self.display.update_display(img_bgr, '3d') # send to display in bgr
        
    def draw_trajectory(self):
        if len(self.trajectory) > 1:
            # TODO: add green squares for the trajectory so we can see the path more clearly
            # remove previous line if exists
            if self.trajectory_line is not None:
                self.vis.remove_geometry(self.trajectory_line)
            
            # create lineset
            points = np.asarray(self.trajectory)
            lines = [[i, i + 1] for i in range(len(self.trajectory) - 1)]
            
            colors = [[1.0, 0.0, 0.0] for _ in range(len(lines))]  # red [r,g,b]
            
            self.trajectory_line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            self.trajectory_line.colors = o3d.utility.Vector3dVector(colors)
            
            # add to visualizer
            self.vis.add_geometry(self.trajectory_line)

    def save_render(self):
        # saves and open point cloud generated when program finishes
        o3d.io.write_point_cloud("output_map.ply", self.pcd)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="3D Map Render", width=960, height=540)
        _vis_settings(vis)
        vis.add_geometry(self.pcd)
        if self.trajectory_line is not None:
            vis.add_geometry(self.trajectory_line)
        vis.run()
        vis.destroy_window()

    def close(self):
        self.save_render()