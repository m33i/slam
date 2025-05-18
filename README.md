## dependencies
```bash
pip install numpy opencv-contrib-python open3d
```

## usage

```bash
#examples
# focal length 500 and 3d with colors
F=500 COLORS=1 python3 slam.py video_examples/test_kitti984.mp4

# focal length 270 (default) open3d window detached and 3d in rgb green
O3D_OUT=1 python3 slam.py video_examples/test_kitti984.mp4
```
### options

`F=n`  changes focal length (default is 270)

`COLORS=1`  3d generation in colors (default is green)

`O3D_OUT=1`  gets open3d window outside the main display for better visualization (default is 0)

`DETECTOR=ORB`  changes feature detector to orb (default is goodFeaturesToTrack -> GFTT + BRIEF decriptor)

### WSL2 Fix
```bash
export XDG_SESSION_TYPE=x11
```