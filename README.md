## Dependencies
```bash
pip install numpy opencv-contrib-python open3d
```

## Usage

```bash
#examples
# focal length 500 and 3d with colors
F=500 COLORS=1 python3 slam.py video_examples/test_kitti984.mp4

# focal length 270 (default) open3d window detached and 3d in rgb green
O3D_OUT=1 python3 slam.py video_examples/test_kitti984.mp4
```
### Options

`F=n`  Changes focal length (default is 270), 500 works much better most of the time

`COLORS=1`  Point cloud generated with colors instead of green

`O3D_OUT=1`  Gets open3d window outside the main display for better visualization

`DETECTOR=ORB`  Changes feature detector to orb (default is goodFeaturesToTrack -> GFTT + BRIEF descriptor)

`SPEED=O.5`  Changes video playback speed from 1 to 0.5; 2.0 = double speed

`KF=1`  Will show keyframes as red squares

`F_MASK` and `SKY_AUTO`
These options create a mask to avoid extracting features from certain areas, such as clouds in the sky, but you can use them for any region. If you set F_MASK to a positive value like 0.7, the system will use only the bottom 70% of the image for feature detection; if you set F_MASK to a negative value like -0.3, it will use only the top 30%. By default, the whole image is used. If SKY_AUTO is set to 1, the system will automatically restrict feature detection to the bottom 60% if it detects features in the upper part. You can always adjust the region manually with F_MASK, regardless of whether auto mode is enabled.

### WSL2 Fix
```bash
export XDG_SESSION_TYPE=x11
```