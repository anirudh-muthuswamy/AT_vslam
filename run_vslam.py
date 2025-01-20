from vslam.apriltag_visual_slam import AprilTagSLAM
from calibration.camera_calibration import load_calibration_images, find_corners_and_calibrate
import os

if __name__ == "__main__":
    calibration_dir = "data/calibration_images"
    image_paths = load_calibration_images(calibration_dir)
    camera_matrix, dist_coeffs = find_corners_and_calibrate(image_paths)

    print("Camera Matrix:", camera_matrix)
    print("Distortion Coefficients:", dist_coeffs)

    image_dir = "data/vslam"
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    print(len(image_paths))

    slam = AprilTagSLAM(camera_matrix)
    tag_keys = slam.build_factor_graph(image_paths)
    result = slam.optimize()
    graph = slam.get_graph()

    print("Initial Error: ", graph.error(slam.initial_estimate))
    print("Final Error: ", graph.error(result))

    slam.plot_camera_and_tag_poses(result, tag_keys, image_paths)
    print("Optimization complete.")
