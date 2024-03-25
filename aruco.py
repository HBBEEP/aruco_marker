import cv2
from utils import ARUCO_DICT
import matplotlib.pyplot as plt
import numpy as np
import yaml

class Aruco():
    def __init__(self):
        self.load_calibration()
        self.rvec, self.tvec = None, None

    def generate(self, aruco_type:str = "DICT_6X6_250", id:int = 0, img_size:int = 700, path:str = "aruco_pic/aruco", plot:bool = True):
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, img_size)
        type_suffix = '_'.join(aruco_type.split('_')[-2:])
        cv2.imwrite(f"{path}_{type_suffix}_{id}.png", marker_img)

        if (plot):
            plt.imshow(marker_img, cmap='gray', interpolation="nearest")
            plt.axis("off")
            plt.show()

    def aruco_display(corners, ids, rejected, image):
    
        if len(corners) > 0:
            
            ids = ids.flatten()
            
            for (markerCorner, markerID) in zip(corners, ids):
                
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID: {}".format(markerID))
                
        return image

    def pose_estimation(self, frame, aruco_dict_type):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)
            
        if len(corners) > 0:
            for i in range(0, len(ids)):
                self.rvec, self.tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.intrinsic_camera, self.distortion)
                cv2.aruco.drawDetectedMarkers(frame, corners) 
                cv2.drawFrameAxes(frame, self.intrinsic_camera, self.distortion, self.rvec, self.tvec, 0.01) 

        return frame

    def get_transformation_matrix(self):
        return  self.tvec, self.rvec
    
    def load_calibration(self):
        # self.intrinsic_camera = np.array(((933.15867, 0, 657.59),(0,933.1586, 400.36993),(0,0,1)))
        # self.distortion = np.array((-0.43948,0.18514,0,0))
        # print(self.intrinsic_camera, self.distortion)
        
        with open('config\calibration.yaml', 'r') as stream:
            data = yaml.safe_load(stream)

            # Load intrinsic camera parameters
            intrinsic_camera_data = data['intrinsic_camera']
            self.intrinsic_camera = np.array(intrinsic_camera_data)

            # Load distortion parameters
            distortion_data = data['distortion'][0]
            self.distortion = np.array(distortion_data)

        print(self.intrinsic_camera, self.distortion)
