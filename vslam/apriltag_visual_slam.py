import apriltag
import gtsam
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


class AprilTagSLAM:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.detector = apriltag.Detector()
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.sigma = 15
        self.noise = gtsam.noiseModel.Isotropic.Sigma(6, self.sigma)

    def detect_tags(self, image_path):
        gray_frame = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray_frame)

    def build_factor_graph(self, image_paths):
        fx, fy = self.camera_matrix[0][0], self.camera_matrix[1][1]
        cx, cy = self.camera_matrix[0][2], self.camera_matrix[1][2]
        
        sorted_paths = sorted(image_paths, key=lambda x: int(x.split('_')[1].split('.')[0]))
        tag_keys = []

        for (i, image_path) in tqdm(enumerate(sorted_paths)):
            detections = self.detect_tags(image_path)
                
            for detection in detections:
                tag_id = detection.tag_id
                if tag_id == 0:
                    tag_key = gtsam.symbol('Y', tag_id)
                    tag_keys.append(tag_key)
                    if tag_key not in self.initial_estimate.keys():
                        #constain pose of tag 0 to origin using prior factor
                        prior_noise = gtsam.noiseModel.Constrained.All(6)
                        self.graph.add(gtsam.PriorFactorPose3(gtsam.symbol('Y', 0), gtsam.Pose3(), prior_noise))
                        self.initial_estimate.insert(tag_key, gtsam.Pose3())
                        X_WT = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
                if tag_id > 0:
                    X_WTi = np.dot(X_WC, X_CT)
                    tag_key = gtsam.symbol('Y', tag_id)
                    tag_keys.append(tag_key)
                    if tag_key not in self.initial_estimate.keys():
                        self.initial_estimate.insert(gtsam.symbol('Y', tag_id), gtsam.Pose3(X_WTi))
                
                X_CT = self.detector.detection_pose(detection, camera_params=(fx, fy, cx, cy))[0]
                X_CT[:3,3] = X_CT[:3,3] * 0.01
                #X_CT (Pose of tag with respect to camera)
                X_CT_rotation = X_CT[:3,:3]
                X_CT_translation = X_CT[:3,3]
                camera_key = gtsam.symbol('X', i)

                if i == 0 and tag_id == 0:
                    X_WC = np.dot(X_WT, np.linalg.inv(X_CT))
                    translation = gtsam.Point3(0, 0, )
                    self.graph.add(gtsam.PriorFactorPose3(gtsam.symbol('X', 0), gtsam.Pose3(X_WC), prior_noise))
                    self.initial_estimate.insert(camera_key,gtsam.Pose3(X_WC))

                if tag_id > 0:
                    if camera_key not in self.initial_estimate.keys():
                        X_WC = np.dot(np.linalg.inv(X_CT),X_WTi)
                        self.initial_estimate.insert(camera_key,gtsam.Pose3(X_WC))

                measurement = gtsam.Pose3(gtsam.Rot3(X_CT_rotation), gtsam.Point3(X_CT_translation))
                self.graph.add(gtsam.BetweenFactorPose3(camera_key, tag_key, measurement, self.noise))
        return tag_keys

    def get_graph(self):
        return self.graph
    
    def plot_camera_and_tag_poses(self, result, tag_keys, image_paths):
        camera_pos = []
        for i in range(0,len(image_paths)):
            camera_pos_sym = gtsam.symbol('X', i)
            optimizedCameraPose = result.atPose3(camera_pos_sym)
            camera_pos.append(optimizedCameraPose.translation())
        camera_pos_x, camera_pos_y, camera_pos_z = zip(*camera_pos)

        tag_pos = []
        for tagPose in tag_keys:
            optimizedTagPose = result.atPose3(tagPose)
            tag_pos.append(optimizedTagPose.translation())
        tag_pos_x, tag_pos_y, tag_pos_z = zip(*tag_pos)

        fig = plt.figure()
        #set axes using tag positions
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.scatter(camera_pos_x, camera_pos_y, camera_pos_z, color='blue')
        ax.scatter(tag_pos_x, tag_pos_y, tag_pos_z, color='red')
        plt.show()

    def optimize(self):
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        return optimizer.optimize()