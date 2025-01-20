import os
import cv2
import numpy as np

def load_calibration_images(directory: str):
    return [os.path.join(directory, img) for img in os.listdir(directory)]

def find_corners_and_calibrate(image_paths, chessboard_size=(8, 6), square_size=0.01):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    observed_pts = []
    world_pts = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img, chessboard_size, None)

        if ret:
            observed_pts.append(corners)
            world_pts.append(objp)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        world_pts, np.array(observed_pts), gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs