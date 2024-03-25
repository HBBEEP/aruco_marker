import cv2
import numpy as np
import yaml
import os
import time


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def camera_calibration(folder_path: str = "calibration_pic", save_path: str = "config/calibration.yaml"):
    square_size = 0.024
    height = 8
    width = 6
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size
    #* 0.025
    # * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    img_folder = os.listdir(folder_path)
    for fname in img_folder:
        image_path = os.path.join(folder_path, fname)  # Construct full path to image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    #         # Draw and display the corners
            cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    # mtx  : "calibration_matrix"
    # dist : "distortion_coefficients" 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    calibration_data = {  
            'intrinsic_camera': mtx.tolist(),
            'distortion': dist.tolist()
        }    
    save_to_yaml_file(save_path, calibration_data)

def image_capture(img_num:int, save_path:str = "calibration_pic"):
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(0)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    count = 0

    while count < img_num:
        ret, frame = cap.read()
        if ret:
            filename = os.path.join(save_path, f"image_{count}.png")
            cv2.imwrite(filename, frame)
            count += 1

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1000)
            if key & 0xFF == ord('q'):
                break
			
        else:
            print("Error capturing image!")
            break

    cap.release()

def save_to_yaml_file(save_path:str, data):
    print(data)
    with open(save_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
        print(f"Data saved to {save_path}")
     
