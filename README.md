# Kamayuc Perception

Perception stack for the exploration task of the Kamayuc/Leo rover.

## Launching

## ROS API

### 1. Depth - RGB Images

### Subscribed topics

* **`rgb/camera_info`** ([sensor_msgs/CameraInfo])
    
    Calibration data for the rgb camera.

* **`rgb/image`** ([sensor_msgs/Image])
    
    Rectified color images from the rgb camera.

* **`depth/camera_info`** ([sensor_msgs/CameraInfo])
    
    Calibration data for the depth camera.

* **`depth/image`** ([sensor_msgs/Image])
    
    Depth map image registered on depth camera image.

### Published topics

## TODO

- [ ] Implement `ARUco` marker detector.
- [ ] Calibrate `alvar.yaml` that stores artag and detection parameters.
- [ ] Show ARTag detection on image, filled the pixels.
- [ ] Implement here visual odometry packages [Visual Odom](https://github.com/klintan/vo-survey).


[sensor_msgs/CameraInfo]: http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html
[sensor_msgs/Image]: http://docs.ros.org/api/sensor_msgs/html/msg/Image.html
