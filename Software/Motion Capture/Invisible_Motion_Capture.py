import cv2
import numpy as np
import time
import socket
import math

# debugging options
# escapeWithKeys: if true, you can stop the script by pressing 'q'
#                 if false, no window will be shown, but you can still stop it with ctrl + c
escapeWithKeys = True
# showProcessingSteps: if true, shows intermediate image processings steps in separate windows
showProcessingSteps = False

# setup to detect ArUco markers
aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()

# set up the camera
video = cv2.VideoCapture(0)

# set up the camera resolution
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort_Right = ("127.0.0.2", 5051)

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.017

# Calibration parameters yaml file
camera_calibration_parameters_filename = 'calibration_chessboard.yaml'

UDP_IP_ADDRESS = "127.0.0.2"
UDP_PORT_NO = 5051
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
sensitive_factor_z = 1000

# function for when slider is adjusted by user
def on_change(value):
    global threshold_val
    threshold_val = value
    print(value)

cv2.namedWindow('Detection Result')
#cv2.namedWindow('Camera stream')
#cv2.createTrackbar('slider', 'Detection Result', 45, 255, on_change)
#cv2.setTrackbarPos('slider', 'Detection Result', threshold_val)
#cv2.namedWindow("Camera stream", cv2.WINDOW_NORMAL)
cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
threshold_val= 31

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
        # print(i, ". sqrt: ", sqrt)

        if sqrt < threshold:
            # min_diff = diff
            index = i
            #print("YES!! 1")

    return index

    # if min_diff < threshold:
    #     return index
    # return None


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
        #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        # calculate the smallest marker size necessary
        smallestSize = frame.shape[0]*frame.shape[1]/700 #500

        # store a copy of the frame for drawing in color later
        frameColor = frame

        # convert it to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Stream', frame)

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

        markerListPrevious = markerListCurrent
        markerListCurrent = []

        print("markerListPrevious: ", markerListPrevious)

        for contour in contours:
            if cv2.contourArea(contour) > smallestSize:
                largest = contour
                # simplify contours
                epsilon = 0.1*cv2.arcLength(contour,True)
                approx = cv2.approxPolyDP(contour,epsilon,True)
                # check if approximation has 4 sides
                if len(approx)==4:
                    frameColor = cv2.drawContours(frameColor, [approx], 0, (100,100,255), 1)
                    print("simplified contour has", len(approx), "points")

                    # get the bounding rectangle
                    x,y,w,h = cv2.boundingRect(contour)

                    ### check if this marker exists in the previously detected marker array
                    if (markerIndex := findMatchingMarker(approx, markerListPrevious, similarityThreshold)) is not None:
                        print("YES!! 2")
                        # this marker was previously decoded, let's use the previous result
                        (markerID, previousMarkerCorners) = markerListPrevious[markerIndex]

                        print("previousMarkerCorners[0][0][0]: ", previousMarkerCorners[0][0])
                        # match the 0th point of aruco to one of the mask corners
                        zerothCornerIndex = findMatchingCorner(previousMarkerCorners[0][0], approx)
                        # reorder the list based on aruco's first corner
                        print("zerothCornerIndex: ", zerothCornerIndex)

                        print("approx: ", approx)
                        # print("originalMask[zerothCornerIndex:]: ", originalMask[zerothCornerIndex:])
                        # print("originalMask[:zerothCornerIndex]: ", originalMask[:zerothCornerIndex])
                        # approx = originalMask[zerothCornerIndex:] + originalMask[:zerothCornerIndex]

                        approx = np.concatenate((approx[zerothCornerIndex:], approx[:zerothCornerIndex]), axis=0)
                        print("approx: ", approx)

                        # reorder approx's points to match the previously ordered list
                        markerListCurrent.append((markerID, approx))
                        cv2.putText(frameColor, "id="+str(markerID), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2, cv2.LINE_AA)

                        ## attempt aruco detection
                        #corners, ids, rejected_img_points = aruco.detectMarkers(frameColor, aruco_dict,
                        #                                                        parameters=parameters)
                        ## if aruco codes are found
                        #if np.all(ids is not None):
                        #
                        #    # Get the rotation and translation vectors
                        #    rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                        #        corners,
                        #        aruco_marker_side_length,
                        #        mtx,
                        #        dst)
                        #
                        #    # Print the pose for the ArUco marker
                        #    # The pose of the marker is with respect to the camera lens frame.
                        #    # Imagine you are looking through the camera viewfinder,
                        #    # the camera lens frame's:
                        #    # x-axis points to the right
                        #    # y-axis points straight down towards your toes
                        #    # z-axis points straight ahead away from your eye, out of the camera
                        #    for i, marker_id in enumerate(ids):
                        #        # Store the translation (i.e. position) information
                        #        transform_translation_x = tvecs[i][0][0]
                        #        transform_translation_y = tvecs[i][0][1]
                        #        transform_translation_z = tvecs[i][0][2]
                        #
                        #        # Draw the axes on the marker
                        #        cv2.aruco.drawAxis(frameColor, mtx, dst, rvecs[i], tvecs[i], 0.05)

                        print("approx[0]: ", approx[0])
                        frameColor = cv2.circle(frameColor, approx[0][0], 2, (100, 100, 255), 3)
                        # no need to run detection on this marker
                        continue

                    ### crop the marker
                    # print("w: ", w)
                    # pad = 10
                    pad = int(w/8)
                    sampleCroppedMarker = frame[y-pad:y+h+pad, x-pad:x+w+pad]
                    # make a copy for later (QR)
                    sampleCroppedMarkerAdaptive = sampleCroppedMarker.copy()

                    originalMask = approx # for later

                    # apply a mask using this contour
                    # first apply the offset to the "approx" contour too
                    x_offset, y_offset = x-pad, y-pad
                    approx = approx - (x_offset, y_offset)
                    # draw filled contour on black background
                    mask = np.zeros_like(sampleCroppedMarker, dtype=np.uint8)
                    #if mask is None:

                    print("MASK ", mask.shape)
                    if mask.shape[0] == 0:
                        continue

                    cv2.drawContours(mask, [approx], 0, (255,255,255), -1)
                    # apply the "approx" mask to marker image
                    sampleCroppedMarker = cv2.bitwise_and(sampleCroppedMarker, mask)

                    # check if mask worked
                    if sampleCroppedMarker is None or sampleCroppedMarker.shape[0]==0:
                        continue

                    # resize cropped marker
                    markerSize = 50 #pixels
                    markerSize = markerSize / sampleCroppedMarker.shape[0] # markerSize = ratio
                    sampleCroppedMarker = cv2.resize(sampleCroppedMarker, (0,0), fx=markerSize, fy=markerSize) 

                    clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8,8))
                    sampleCroppedMarker = clahe.apply(sampleCroppedMarker)

                    # _, sampleCroppedMarker = cv2.threshold(sampleCroppedMarker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # prepare for aruco detection
                    sampleCroppedMarker = cv2.bitwise_not(sampleCroppedMarker)
                    image_drawn_aruco = cv2.cvtColor(sampleCroppedMarker,cv2.COLOR_GRAY2BGR)

                    # attempt aruco detection
                    corners, ids, rejected_img_points = aruco.detectMarkers(sampleCroppedMarker, aruco_dict, parameters=parameters)
                    
                    # if aruco codes are found
                    if np.all(ids is not None):
                        aruco.drawDetectedMarkers(image_drawn_aruco, corners)


                        for id in ids:

                            id = id[0]
                            # print(id)
                            # print("markersCounted: ", markersCounted)
                            markersCounted = markersCounted + 1

                            #cv2.putText(frameColor, "id="+str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 2, cv2.LINE_AA)

                            # store the detected marker for future frames
                            # match the 0th point of aruco to one of the mask corners
                            approx = approx * markerSize
                            zerothCornerIndex = findMatchingCorner(corners[0][0][0], approx)
                            # reorder the list based on aruco's first corner
                            print("zerothCornerIndex: ", zerothCornerIndex)

                            print("originalMask: ", originalMask)
                            # print("originalMask[zerothCornerIndex:]: ", originalMask[zerothCornerIndex:])
                            # print("originalMask[:zerothCornerIndex]: ", originalMask[:zerothCornerIndex])
                            # originalMask = originalMask[zerothCornerIndex:] + originalMask[:zerothCornerIndex]

                            originalMask = np.concatenate((originalMask[zerothCornerIndex:], originalMask[:zerothCornerIndex]), axis=0)

                            print("originalMask 2: ", originalMask)

                            # print("approx: ", approx)
                            # print("corners[0][0][0]: ", corners[0][0][0])
                            
                            markerListCurrent.append((id, originalMask))

                    if showProcessingSteps:
                        cv2.imshow('ArUco image preprocessing', image_drawn_aruco)

        ### Calculating the FPS
        fps = 1 / (time.time() - prev_frame_time)
        print("time: ", (time.time() - prev_frame_time))
        print("fps: ", fps)
        # converting the fps into integer
        fps = int(fps)

        # draw the FPS as text
        cv2.putText(frameColor, str(fps), (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        print("markerListCurrent: ", markerListCurrent)

        # Unity part
        markers_to_unity = []
        markers_left = []
        markers_right = []

        left_x = int(0)
        left_y = int(0)
        left_z = int(0)
        right_x = int(0)
        right_y = int(0)
        right_z = int(0)

        for marker in markerListCurrent:
            #print("MARKER ID: ", marker[0])
            #print("MARKER CORNERS: ", marker[1])
            #marker_converted = marker[1].astype('float32') # sorry Doga, Raul's ChatGPT wons this battle
            marker_converted = marker[1].astype(np.float32).reshape((1, 4, 2))

            # Get the rotation and translation vectors
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                marker_converted,
                aruco_marker_side_length,
                mtx,
                dst)

            # Store the translation (i.e. position) information
            transform_translation_x = tvecs[0][0][0]
            transform_translation_y = tvecs[0][0][1]
            transform_translation_z = tvecs[0][0][2]

            ## from github https://github.com/opencv/opencv/issues/8813
            T = tvecs[0, 0]
            R = cv2.Rodrigues(rvecs[0])[0]
            # Unrelated -- makes Y the up axis, Z forward
            R = R @ np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0],
            ])
            if 0 < R[1, 1] < 1:
                # If it gets here, the pose is flipped.

                # Flip the axes. E.g., Y axis becomes [-y0, -y1, y2].
                R *= np.array([
                    [1, -1, 1],
                    [1, -1, 1],
                    [-1, 1, -1],
                ])

                # Fixup: rotate along the plane spanned by camera's forward (Z) axis and vector to marker's position
                forward = np.array([0, 0, 1])
                tnorm = T / np.linalg.norm(T)
                axis = np.cross(tnorm, forward)
                angle = -2 * math.acos(tnorm @ forward)
                R = cv2.Rodrigues(angle * axis)[0] @ R

            marker_to_unity = [marker[0], int(transform_translation_y * sensitive_factor_y), int(transform_translation_x * sensitive_factor_x),
                       int(transform_translation_z * sensitive_factor_z)]

            markers_to_unity.append(marker_to_unity)
            print('==========================')
            print('RAUL UNITY =', markers_to_unity)

            # Draw the axes on the marker
            #cv2.aruco.drawAxis(frameColor, mtx, dst, rvecs[0], tvecs[0], 0.05)

            cv2.drawFrameAxes(frameColor, mtx, dst, R, tvecs[0], aruco_marker_side_length / 2)

            # =========================================================
            # Split the coordinates into right and left hand
            # =========================================================
            for i in markers_to_unity:
                if i[0] / 10 < 1:
                    #print(i[0])
                    markers_left.append(i)
                    left_x += i[1]
                    left_y += i[2]
                    left_z += i[3]

                elif i[0] / 10 >= 1:
                    #print(i[0])
                    markers_right.append(i)
                    right_x += i[1]
                    right_y += i[2]
                    right_z += i[3]

                else:
                    pass

            # ==================================================
            # Alpha-beta filter
            # ==================================================

            try:
                left_x_final = alpha * (left_x / len(markers_left)) + (1 - alpha) * left_x_final
                left_y_final = alpha * (left_y / len(markers_left)) + (1 - alpha) * left_y_final
                left_z_final = alpha * (left_z / len(markers_left)) + (1 - alpha) * left_z_final

            except:
                # print('qooqoo_left')
                pass

            try:
                right_x_final = alpha * (right_x / len(markers_right)) + (1 - alpha) * right_x_final
                right_y_final = alpha * (right_y / len(markers_right)) + (1 - alpha) * right_y_final
                right_z_final = alpha * (right_z / len(markers_right)) + (1 - alpha) * right_z_final
            except:
                # print('qooqoo_right')
                pass


            # Send the coordinates to Unity
            #Message = str('['F"{int(left_x_final)},{int(left_y_final)},{int(left_z_final)},{int(right_x_final)},{int(right_y_final)},{int(right_z_final)}"']')
            Message = str('['F"{int(left_x_final)},{int(left_y_final)},{int(left_z_final)},{int(right_x_final)},{int(right_y_final)},{int(right_z_final)},{int(marker[0])}"']')

            sock.sendto(str.encode(str(Message)), serverAddressPort_Right)

        #cv2.resizeWindow('Detection Result', 533, 400)
        #cv2.namedWindow("Camera stream", cv2.WINDOW_NORMAL)
        cv2.imshow('Detection Result', frameColor)

        #output.write(final)
        
        if escapeWithKeys:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    else:
        break

print("markersCounted: ", markersCounted)

video.release()
cv2.destroyAllWindows()