import open3d as o3d
import numpy as np

class Mapping:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        # 3d window pos is set to the right
        self.vis.create_window(window_name="3D Map", width=960, height=540, left=1000)
        
        # initialize an empty point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # set the background color to black
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        
        # initial view
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])

    def update_map(self, points, colors=None):
        if len(points) == 0:
            return

        # convert points to Open3D format point Cloud
        points = np.asarray(points)
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            current_pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            current_pcd.paint_uniform_color([1, 1, 1])

        # merge with existing point cloud
        if len(self.pcd.points) == 0:
            self.pcd.points = current_pcd.points
            self.pcd.colors = current_pcd.colors
        else:
            self.pcd.points = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(self.pcd.points), np.asarray(current_pcd.points)))
            )
            self.pcd.colors = o3d.utility.Vector3dVector(
                np.vstack((np.asarray(self.pcd.colors), np.asarray(current_pcd.colors)))
            )

        # update
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()