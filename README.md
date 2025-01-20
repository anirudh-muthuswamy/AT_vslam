# Overview

This repository implements a Visual Simultaneous Localization and Mapping (VSLAM) system using AprilTags and the GTSAM optimization library. 
The project enables robust camera pose estimation and mapping by leveraging AprilTag detection and graph-based optimization techniques.

# Features

- Camera Calibration: Precise camera intrinsic parameter estimation using chessboard calibration
- AprilTag Detection: Robust tag detection and pose estimation
- Graph-Based Optimization: Pose graph optimization using GTSAM's Levenberg-Marquardt algorithm
- Visualization: 3D visualization of camera and tag poses
  
# Dependencies

- OpenCV - 4.8.0.74
- NumPy - 1.26.4
- AprilTag - 0.0.16
- GTSAM - 4.2
- Matplotlib - 3.9.2
