import cv2
import numpy as np
import time
from pyzbar.pyzbar import decode
#from dbr_qrcode import get_dbr_detector
import socket
import math
from scipy.spatial.transform import Rotation as Rscipy

# debugging options
# detectQR: if true, attemps to decode QR codes
detectQR = False
# escapeWithKeys: if true, you can stop the script by pressing 'q'
#                 if false, no window will be shown, but you can still stop it with ctrl + c
escapeWithKeys = True
# showProcessingSteps: if true, shows intermediate image processings steps in separate windows
showProcessingSteps = True

# setup to detect ArUco markers
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
#parameters = aruco.DetectorParameters()
# parameters = aruco.DetectorParameters_create()
parameters =  aruco.DetectorParameters()

# video = cv2.VideoCapture('videos/RAW_IR_Camera_Footage.mov')
# video = cv2.VideoCapture('videos/50fps_850nm Filter_760 IR Light Module_12in Distance_Vertical Motion.avi')
# video = cv2.VideoCapture('videos/Diffused 2x Raw Video.mov')
# video = cv2.VideoCapture('videos/WIN_20230124_14_36_04_Pro.mp4')
# video = cv2.VideoCapture('videos/Wristband Test w LED.mov')
# video = cv2.VideoCapture('videos/Aruco Wristband Full Test.mov')

# video = cv2.VideoCapture(0)
video = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort_Right = ("127.0.0.1", 5055)

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.017

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'

# on Mac, only the 127.0.0.1 IP address works
UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 5055
Message = "0"
clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load the camera parameters from the saved file
cv_file = cv2.FileStorage(
    camera_calibration_parameters_filename, cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

alpha=float(0.7)
left_x_final = 0
left_y_final = 0
left_z_final = 0

right_x_final = 0
right_y_final = 0
right_z_final = 0

sensitive_factor_x = 4000
sensitive_factor_y = 6000
sensitive_factor_z = 2000

# function for when slider is adjusted by user
def on_change(value):
    global threshold_val
    threshold_val = value
    print(value)

cv2.namedWindow('Detection Result')
cv2.createTrackbar('slider', 'Detection Result', 42, 255, on_change)
cv2.setTrackbarPos('slider', 'Detection Result', threshold_val)

# output = cv2.VideoWriter('output_video.mp4v', cv2.VideoWriter_fourcc(*'XVID'), 50, (2048, 1152))


similarityThreshold = 10

def findMatchingCorner(corner, cornerList):
    """
    Returns the index of the point in cornerList that is closest to corner 
    """
    min_diff = float('inf')

    for i, c in enumerate(cornerList):

        print("corner: ", corner)
        print("c: ", c)
        sqrt = np.sqrt(np.sum((corner - c)**2))
        print("sqrt: ", sqrt)

        if sqrt < min_diff:
            print("yes small: ", i)
            min_diff = sqrt
            index = i

    return index


def findMatchingMarker(new_corners, marker_array, threshold=similarityThreshold):
    """
    Find the index of the aruco marker with very similar corners compared to the new marker.
    
    Parameters:
        new_corners (ndarray): The new aruco marker's corner set.
        marker_array (list of tuples): A list of tuples, where each tuple contains the marker ID and corners.
        threshold (float): The threshold value used to define the similarity between the corners.
        
    """
    index = None
    new_marker_center = np.mean(new_corners, axis=0)
    print("new_corners: ", new_corners)
    print("new_marker_center: ", new_marker_center)

    for i, (marker_id, marker_corners) in enumerate(marker_array):
        marker_center = np.mean(marker_corners, axis=0)
        # print("marker_center: ", marker_center)
        sqrt = np.sqrt(np.sum((marker_center - new_marker_center)**2))
        print(i, ". sqrt: ", sqrt)

        if sqrt < threshold:
            # min_diff = diff
            index = i
            #print("YES!! 1")

    return index

    # if min_diff < threshold:
    #     return index
    # return None


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


markersCounted = 0

markerListCurrent = []

# # used to record the time when we processed last frame
# prev_frame_time = 0

# # used to record the time at which we processed current frame
# new_frame_time = 0

while video.isOpened():

    prev_frame_time = time.time()

    ret, frame = video.read()

    # check if a frame exists in the video
    if ret:

        # half the input frame size
        frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)

        # calculate the smallest marker size necessary
        smallestSize = frame.shape[0]*frame.shape[1]/700 #500

        # store a copy of the frame for drawing in color later
        frameColor = frame

        # convert it to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # darken the image: subtract the threshold value
        frame_offsetted = np.double(frame)
        frame_offsetted = frame_offsetted - threshold_val
        frame_offsetted = np.maximum(frame_offsetted, 0)
        frame_offsetted = np.uint8(frame_offsetted)

        # erode the binary blobs slightly
        # kernel = np.ones((threshold_val, threshold_val), np.uint8)
        erosion_size = 2*2 + 1
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))

        frame_offsetted = cv2.erode(frame_offsetted, element) 

        # frame = 
        # print("frame", frame)
        # frameColor = frame_offsetted
        
        if showProcessingSteps:
            cv2.imshow('frame_offsetted', frame_offsetted)

        _, otsu = cv2.threshold(frame_offsetted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if showProcessingSteps:
            cv2.imshow('Binarization', otsu)

        # apply contour detection (4 sides)
        contours, _ = cv2.findContours(otsu,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # other opencv version:
        # _, contours, _ = cv2.findContours(otsu,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # #frameColor = cv2.drawContours(frameColor, contours, 0, (255,255,100), 2)
        # print("# of contours:", len(contours))

        # if len(contours)!=0:
        #     print("1st contour has ", len(contours[0]), "points")

        # markerListPrevious = markerListCurrent
        # markerListCurrent = []

        # print("markerListPrevious: ", markerListPrevious)



        # ## for hololens: skip weird frames
        # largestContour = max(contours, key = cv2.contourArea)
        # largeBlobThreshold = frame.shape[0]*frame.shape[1]/30 #500


        # sort the contours in decreasing order of area
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # select the second largest contour (if it exists)
        if len(sorted_contours) > 1:
            largest_contour0 = sorted_contours[0]
            largest_contour1 = sorted_contours[1]
        else:
            print("There is no second largest contour.")
            cv2.imshow('Detection Result', frameColor)
            continue

        M0 = cv2.moments(largest_contour0)
        if M0["m00"] != 0:
            center0 = (int(M0["m10"] / M0["m00"]), int(M0["m01"] / M0["m00"]))

        M1 = cv2.moments(largest_contour1)
        if M1["m00"] != 0:
            center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))

        # draw a line between the centroids
        line_color = (0, 0, 255) # BGR format (red)
        line_thickness = 2
        cv2.line(frameColor, center0, center1, line_color, line_thickness)

        # compare the y-coordinates of the centroids
        if center0[1] > center1[1]:
            print("largest_contour0 is above ")
            targetCoordinate_y = int((center0[0] - frame.shape[0]/2) * 100 / (frame.shape[0]/2))
            targetCoordinate_x = int((center0[1] - frame.shape[1]/2) * 100 / (frame.shape[1]/2))
            targetCoordinate = (targetCoordinate_x, targetCoordinate_y)
        else:
            print("largest_contour1 is above")
            targetCoordinate_y = int((center1[0] - frame.shape[0]/2) * 100 / (frame.shape[0]/2))
            targetCoordinate_x = int((center1[1] - frame.shape[1]/2) * 100 / (frame.shape[1]/2))
            targetCoordinate = (targetCoordinate_x, targetCoordinate_y)

        #print("largestContour area: ", cv2.contourArea(largestContour))



        # for contour in contours:
        #     if cv2.contourArea(contour) > smallestSize:
        #         largest = contour
        #         # simplify contours
        #         epsilon = 0.1*cv2.arcLength(contour,True)
        #         approx = cv2.approxPolyDP(contour,epsilon,True)
        #         # check if approximation has 4 sides
        #         if len(approx)==4:
        #             frameColor = cv2.drawContours(frameColor, [approx], 0, (100,100,255), 1)
        #             print("simplified contour has", len(approx), "points")

        #             # get the bounding rectangle
        #             x,y,w,h = cv2.boundingRect(contour)

        #             ### check if this marker exists in the previously detected marker array
        #             if (markerIndex := findMatchingMarker(approx, markerListPrevious, similarityThreshold)) is not None:
        #                 print("YES!! 2")
        #                 # this marker was previously decoded, let's use the previous result
        #                 (markerID, previousMarkerCorners) = markerListPrevious[markerIndex]

        #                 print("previousMarkerCorners[0][0][0]: ", previousMarkerCorners[0][0])
        #                 # match the 0th point of aruco to one of the mask corners
        #                 zerothCornerIndex = findMatchingCorner(previousMarkerCorners[0][0], approx)
        #                 # reorder the list based on aruco's first corner
        #                 print("zerothCornerIndex: ", zerothCornerIndex)

        #                 print("approx: ", approx)
        #                 # print("originalMask[zerothCornerIndex:]: ", originalMask[zerothCornerIndex:])
        #                 # print("originalMask[:zerothCornerIndex]: ", originalMask[:zerothCornerIndex])
        #                 # approx = originalMask[zerothCornerIndex:] + originalMask[:zerothCornerIndex]

        #                 approx = np.concatenate((approx[zerothCornerIndex:], approx[:zerothCornerIndex]), axis=0)
        #                 print("approx: ", approx)

        #                 # reorder approx's points to match the previously ordered list
        #                 markerListCurrent.append((markerID, approx))
        #                 cv2.putText(frameColor, "id="+str(markerID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2, cv2.LINE_AA)

        #                 ## attempt aruco detection
        #                 #corners, ids, rejected_img_points = aruco.detectMarkers(frameColor, aruco_dict,
        #                 #                                                        parameters=parameters)
        #                 ## if aruco codes are found
        #                 #if np.all(ids is not None):
        #                 #
        #                 #    # Get the rotation and translation vectors
        #                 #    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
        #                 #        corners,
        #                 #        aruco_marker_side_length,
        #                 #        mtx,
        #                 #        dst)
        #                 #
        #                 #    # Print the pose for the ArUco marker
        #                 #    # The pose of the marker is with respect to the camera lens frame.
        #                 #    # Imagine you are looking through the camera viewfinder,
        #                 #    # the camera lens frame's:
        #                 #    # x-axis points to the right
        #                 #    # y-axis points straight down towards your toes
        #                 #    # z-axis points straight ahead away from your eye, out of the camera
        #                 #    for i, marker_id in enumerate(ids):
        #                 #        # Store the translation (i.e. position) information
        #                 #        transform_translation_x = tvecs[i][0][0]
        #                 #        transform_translation_y = tvecs[i][0][1]
        #                 #        transform_translation_z = tvecs[i][0][2]
        #                 #
        #                 #        # # Draw the axes on the marker
        #                 #        # cv2.aruco.drawAxis(frameColor, mtx, dst, rvecs[i], tvecs[i], 0.05)

        #                 print("approx[0]: ", approx[0])
        #                 frameColor = cv2.circle(frameColor, approx[0][0], 2, (100, 100, 255), 3)
        #                 # no need to run detection on this marker
        #                 continue

        #             ### crop the marker
        #             # print("w: ", w)
        #             # pad = 10
        #             pad = int(w/8)
        #             sampleCroppedMarker = frame[y-pad:y+h+pad, x-pad:x+w+pad]
        #             # make a copy for later (QR)
        #             sampleCroppedMarkerAdaptive = sampleCroppedMarker.copy()

        #             originalMask = approx # for later

        #             # apply a mask using this contour
        #             # first apply the offset to the "approx" contour too
        #             x_offset, y_offset = x-pad, y-pad
        #             approx = approx - (x_offset, y_offset)
        #             # draw filled contour on black background
        #             mask = np.zeros_like(sampleCroppedMarker, dtype=np.uint8)
        #             #if mask is None:

        #             print("MASK ", mask.shape)
        #             if mask.shape[0] == 0:
        #                 continue

        #             cv2.drawContours(mask, [approx], 0, (255,255,255), -1)
        #             # apply the "approx" mask to marker image
        #             sampleCroppedMarker = cv2.bitwise_and(sampleCroppedMarker, mask)

        #             # check if mask worked
        #             if sampleCroppedMarker is None or sampleCroppedMarker.shape[0]==0:
        #                 continue

        #             # resize cropped marker
        #             markerSize = 50 #pixels
        #             markerSize = markerSize / sampleCroppedMarker.shape[0] # markerSize = ratio
        #             sampleCroppedMarker = cv2.resize(sampleCroppedMarker, (0,0), fx=markerSize, fy=markerSize) 

        #             clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8,8))
        #             sampleCroppedMarker = clahe.apply(sampleCroppedMarker)

        #             # _, sampleCroppedMarker = cv2.threshold(sampleCroppedMarker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #             # prepare for aruco detection
        #             sampleCroppedMarker = cv2.bitwise_not(sampleCroppedMarker)
        #             image_drawn_aruco = cv2.cvtColor(sampleCroppedMarker,cv2.COLOR_GRAY2BGR)

        #             # attempt aruco detection
        #             corners, ids, rejected_img_points = aruco.detectMarkers(sampleCroppedMarker, aruco_dict, parameters=parameters)
                    
        #             # if aruco codes are found
        #             if np.all(ids is not None):
        #                 aruco.drawDetectedMarkers(image_drawn_aruco, corners)


        #                 for id in ids:

        #                     id = id[0]
        #                     # print(id)
        #                     # print("markersCounted: ", markersCounted)
        #                     markersCounted = markersCounted + 1

        #                     cv2.putText(frameColor, "id="+str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 2, cv2.LINE_AA)

        #                     # store the detected marker for future frames
        #                     # match the 0th point of aruco to one of the mask corners
        #                     approx = approx * markerSize
        #                     zerothCornerIndex = findMatchingCorner(corners[0][0][0], approx)
        #                     # reorder the list based on aruco's first corner
        #                     print("zerothCornerIndex: ", zerothCornerIndex)

        #                     print("originalMask: ", originalMask)
        #                     # print("originalMask[zerothCornerIndex:]: ", originalMask[zerothCornerIndex:])
        #                     # print("originalMask[:zerothCornerIndex]: ", originalMask[:zerothCornerIndex])
        #                     # originalMask = originalMask[zerothCornerIndex:] + originalMask[:zerothCornerIndex]

        #                     originalMask = np.concatenate((originalMask[zerothCornerIndex:], originalMask[:zerothCornerIndex]), axis=0)

        #                     print("originalMask 2: ", originalMask)

        #                     # print("approx: ", approx)
        #                     # print("corners[0][0][0]: ", corners[0][0][0])
                            
        #                     markerListCurrent.append((id, originalMask))

        #             elif detectQR:
        #                 # no Aruco detected, try QR code too (if detectQR is true)
        #                 # detector.decode_buffer(img) != None

        #                 # resize cropped marker
        #                 markerSize = 100 #pixels
        #                 markerSize = markerSize / sampleCroppedMarkerAdaptive.shape[0]
        #                 sampleCroppedMarkerAdaptive = cv2.resize(sampleCroppedMarkerAdaptive, (0,0), fx=markerSize, fy=markerSize) 

        #                 # invert b<->w
        #                 sampleCroppedMarkerAdaptive = cv2.bitwise_not(sampleCroppedMarkerAdaptive)

        #                 # apply adaptive thresholding
        #                 sampleCroppedMarkerAdaptive = cv2.adaptiveThreshold(sampleCroppedMarkerAdaptive,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #                     cv2.THRESH_BINARY,9,1)

        #                 image_drawn_qr = cv2.cvtColor(sampleCroppedMarkerAdaptive,cv2.COLOR_GRAY2BGR)

        #                 # attempt to detect QR codes
        #                 decoded_list = decode(sampleCroppedMarkerAdaptive)

        #                 if decoded_list!=[]:
        #                     # if QR code is found
        #                     message = decoded_list[0].data.decode()
        #                     print("qr success! ", message)
        #                     cv2.putText(frameColor, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        #                 if showProcessingSteps:
        #                     cv2.imshow('QR image preprocessing', image_drawn_qr)

        #             if showProcessingSteps:
        #                 cv2.imshow('ArUco image preprocessing', image_drawn_aruco)

        # ### Calculating the FPS
        # fps = 1 / (time.time() - prev_frame_time)
        # print("time: ", (time.time() - prev_frame_time))
        # print("fps: ", fps)
        # # converting the fps into integer
        # fps = int(fps)

        # # draw the FPS as text
        # cv2.putText(frameColor, str(fps), (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # print("markerListCurrent: ", markerListCurrent)

        # # Unity part
        # markers_to_unity = []
        # markers_left = []
        # markers_right = []

        # left_x = int(0)
        # left_y = int(0)
        # left_z = int(0)
        # right_x = int(0)
        # right_y = int(0)
        # right_z = int(0)

        # for marker in markerListCurrent:
        #     #print("MARKER ID: ", marker[0])
        #     #print("MARKER CORNERS: ", marker[1])
        #     #marker_converted = marker[1].astype('float32') # sorry Doga, Raul's ChatGPT wons this battle
        #     marker_converted = marker[1].astype(np.float32).reshape((1, 4, 2))

        #     # Get the rotation and translation vectors
        #     rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
        #         marker_converted,
        #         aruco_marker_side_length,
        #         mtx,
        #         dst)

        #     # Store the translation (i.e. position) information
        #     transform_translation_x = tvecs[0][0][0]
        #     transform_translation_y = tvecs[0][0][1]
        #     transform_translation_z = tvecs[0][0][2]


        #     ## from github https://github.com/opencv/opencv/issues/8813
        #     T = tvecs[0,0]
        #     R = cv2.Rodrigues(rvecs[0])[0]
        #     # Unrelated -- makes Y the up axis, Z forward
        #     R = R @ np.array([
        #         [1, 0, 0],
        #         [0, 0, 1],
        #         [0,-1, 0],
        #     ])
        #     if 0 < R[1,1] < 1:
        #         # If it gets here, the pose is flipped.

        #         # Flip the axes. E.g., Y axis becomes [-y0, -y1, y2].
        #         R *= np.array([
        #             [ 1, -1,  1],
        #             [ 1, -1,  1],
        #             [-1,  1, -1],
        #         ])
                
        #         # Fixup: rotate along the plane spanned by camera's forward (Z) axis and vector to marker's position
        #         forward = np.array([0, 0, 1])
        #         tnorm = T / np.linalg.norm(T)
        #         axis = np.cross(tnorm, forward)
        #         angle = -2*math.acos(tnorm @ forward)
        #         R = cv2.Rodrigues(angle * axis)[0] @ R



        #     # Store the rotation information
        #     # rotation_matrix = np.eye(4)
        #     # rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[0][0]))[0]
        #     # r = R.from_matrix(rotation_matrix[0:3, 0:3])
        #     r = Rscipy.from_matrix(R)
        #     quat = r.as_quat()

        #     # Quaternion format
        #     transform_rotation_x = quat[0]
        #     transform_rotation_y = quat[1]
        #     transform_rotation_z = quat[2]
        #     transform_rotation_w = quat[3]

        #     # Euler angle format in radians
        #     roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x,
        #                                                    transform_rotation_y,
        #                                                    transform_rotation_z,
        #                                                    transform_rotation_w)

        #     roll_x = math.degrees(roll_x)
        #     pitch_y = math.degrees(pitch_y)
        #     yaw_z = math.degrees(yaw_z)


        #     marker_to_unity = [marker[0], int(transform_translation_y * sensitive_factor_y), int(transform_translation_x * sensitive_factor_x),
        #                int(transform_translation_z * sensitive_factor_z)]

        #     markers_to_unity.append(marker_to_unity)
        #     print('==========================')
        #     print('RAUL UNITY =', markers_to_unity)

        #     # Draw the axes on the marker

        #     print("rvecs, ", rvecs)

        #     print("rvecs2, ", cv2.Rodrigues(R)[0])
        #     print("rvecs3, ", Rscipy.from_matrix(R))
        #     print("R, ", R)
        #     # cv2.drawFrameAxes(frameColor, mtx, dst, rvecs, tvecs[0], aruco_marker_side_length/2)
        #     cv2.drawFrameAxes(frameColor, mtx, dst, R, tvecs[0], aruco_marker_side_length/2)
        #     # y is green
        #     #cv2.drawFrameAxes(frameColor, mtx, dst, cv2.Rodrigues(R)[0], tvecs[0], aruco_marker_side_length/2)

        #     # =========================================================
        #     # Split the coordinates into right and left hand
        #     # =========================================================
        #     for i in markers_to_unity:
        #         if i[0] / 10 < 1:
        #             #print(i[0])
        #             markers_left.append(i)
        #             left_x += i[1]
        #             left_y += i[2]
        #             left_z += i[3]

        #         elif i[0] / 10 >= 1:
        #             #print(i[0])
        #             markers_right.append(i)
        #             right_x += i[1]
        #             right_y += i[2]
        #             right_z += i[3]

        #         else:
        #             pass

        #     # ==================================================
        #     # Alpha-beta filter
        #     # ==================================================

        #     try:
        #         left_x_final = alpha * (left_x / len(markers_left)) + (1 - alpha) * left_x_final
        #         left_y_final = alpha * (left_y / len(markers_left)) + (1 - alpha) * left_y_final
        #         left_z_final = alpha * (left_z / len(markers_left)) + (1 - alpha) * left_z_final

        #     except:
        #         # print('qooqoo_left')
        #         pass

        #     try:
        #         right_x_final = alpha * (right_x / len(markers_right)) + (1 - alpha) * right_x_final
        #         right_y_final = alpha * (right_y / len(markers_right)) + (1 - alpha) * right_y_final
        #         right_z_final = alpha * (right_z / len(markers_right)) + (1 - alpha) * right_z_final
        #     except:
        #         # print('qooqoo_right')
        #         pass


        #     # Send the coordinates to Unity
            # roll_x, pitch_y, yaw_z


        #Message = str('['F"{int(left_x_final)},{int(left_y_final)},{int(left_z_final)},{int(right_x_final)},{int(right_y_final)},{int(right_z_final)}"']')
        Message = str('['F"{targetCoordinate[0]},{targetCoordinate[1]}"']')
        #Message = '[' + str(targetCoordinate) + ']'

        sock.sendto(str.encode(str(Message)), serverAddressPort_Right)



        print("message sent! ", Message)

        cv2.imshow('Detection Result', frameColor)

        #output.write(final)
        
        if escapeWithKeys:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    else:
        
        # reached the end of the video
        # break

        # print('no video')
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

print("markersCounted: ", markersCounted)

video.release()
cv2.destroyAllWindows()
