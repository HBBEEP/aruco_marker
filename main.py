from aruco import Aruco
from utils import ARUCO_DICT, FRAME_WIDTH, FRAME_HEIGHT
import cv2
TIME_STEP = 100

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

aruco_type = "DICT_6X6_250"
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()

aruco_marker = Aruco()


while cap.isOpened():
    ret, frame = cap.read()

    # if (mode == 0):
    # tvec, rvec = aruco_marker.get_transformation_matrix()
    # print(tvec, rvec)

    corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    
    frame = aruco_marker.aruco_display(corners, ids, rejected, frame)

    frame = aruco_marker.pose_estimation(frame, ARUCO_DICT[aruco_type])
    # bbox_width = 1280 // 2
    # bbox_height = 720 // 2

    # x1 = (1280 - bbox_width) // 2
    # y1 = (720 - bbox_height) // 2
    # x2 = x1 + bbox_width
    # y2 = y1 + bbox_height
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Aruco Detector', frame)


    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()