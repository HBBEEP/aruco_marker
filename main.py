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
    frame_out = aruco_marker.pose_estimation(frame, ARUCO_DICT[aruco_type])
    tvec, rvec = aruco_marker.get_transformation_matrix()
    print(tvec, rvec)
    # Calculate the size of the bounding box
    bbox_width = 1280 // 2
    bbox_height = 720 // 2

    # Calculate the coordinates of the bounding box
    x1 = (1280 - bbox_width) // 2
    y1 = (720 - bbox_height) // 2
    x2 = x1 + bbox_width
    y2 = y1 + bbox_height

    # Draw the bounding box on the frame
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('Aruco Detector', frame_out)

    # if (tvec and rvec is not None):
    #     print(f"Translation : {tvec}")
    #     print(f"Rotation : {rvec}")

    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()