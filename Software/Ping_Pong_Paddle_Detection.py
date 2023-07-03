import cv2
import numpy as np
import time
import threading
import autoit

detectQR = False
escapeWithKeys = True
showProcessingSteps = True

aruco = cv2.aruco
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
aruco_marker_side_length = 0.017
x, y = 0, 0
x_scale, y_scale = np.asarray((3840, 2160)) / (np.asarray((1600, 1200)) * np.asarray((0.3, 0.3)))

video = cv2.VideoCapture(1 + cv2.CAP_DSHOW)

cv_file = cv2.FileStorage(
    'calibration_chessboard.yaml', cv2.FILE_STORAGE_READ)
mtx = cv_file.getNode('K').mat()
dst = cv_file.getNode('D').mat()
cv_file.release()

# Thread Mouse Movements
def thread_function(parameter):
    global x
    global y
    while True:
     autoit.mouse_move(int(x * x_scale), int(y * y_scale))
t = threading.Thread(target=thread_function, args=(0,))
t.start()

# Change Threshold Value
def on_change(value):
    global threshold_val
    threshold_val = value

# Build Display Window
cv2.namedWindow('Detection Result')
cv2.createTrackbar('slider', 'Detection Result', 16, 255, on_change)
cv2.setTrackbarPos('slider', 'Detection Result', threshold_val)


def findMatchingCorner(corner, cornerList):
    """
    Returns the index of the point in cornerList that is closest to corner 
    """
    min_diff = float('inf')
    for i, point in enumerate(cornerList):
        sqrt = np.sqrt(np.sum((corner - point)**2))
        if sqrt < min_diff:
            min_diff = sqrt
            index = i
    return index


def findMatchingMarker(new_corners, marker_array, threshold=10):
    """
    Find the index of the aruco marker with very similar corners compared to the new marker.
    
    Parameters:
        new_corners (ndarray): The new aruco marker's corner set.
        marker_array (list of tuples): A list of tuples, where each tuple contains the marker ID and corners.
        threshold (float): The threshold value used to define the similarity between the corners.
        
    """
    index = None
    new_marker_center = np.mean(new_corners, axis=0)
    for i, (marker_id, marker_corners) in enumerate(marker_array):
        marker_center = np.mean(marker_corners, axis=0)
        sqrt = np.sqrt(np.sum((marker_center - new_marker_center)**2))
        if sqrt < threshold:
            index = i
    return index


markersCounted = 0
markerListCurrent = []
while video.isOpened():
    prev_frame_time = time.time()

    ret, frame = video.read()

    if ret:
        # Compress/Shape Frame
        frame = cv2.resize(frame, (0,0), fx=0.4, fy=0.4)
        smallestSize = frame.shape[0]*frame.shape[1]/700
        frameColor = frame
        # Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Thresholding
        frame_offsetted = np.uint8(np.maximum(np.double(frame) - threshold_val, 0))
        # Erode Binary Blobs
        frame_offsetted = cv2.erode(frame_offsetted, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) 
        # Binarization
        _, otsu = cv2.threshold(frame_offsetted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Contour Detection
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if showProcessingSteps:
            cv2.imshow('frame_offsetted', frame_offsetted)

        if showProcessingSteps:
            cv2.imshow('Binarization', otsu)

        markerListCurrent = []
        for contour in contours:
            if cv2.contourArea(contour) > smallestSize:
                # Build bounding box
                largest = contour
                epsilon = 0.1 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4:
                    frameColor = cv2.drawContours(frameColor, [approx], 0, (100,100,255), 2)
                    # Retrieve bounding box dimensions
                    x, y, w, h = cv2.boundingRect(contour)
                    x = x+w/2
                    y = y+h/2

        # Display FPS
        # fps = 60 # int(1 / (time.time() - prev_frame_time))
        # cv2.putText(frameColor, str(fps), (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # Display Detection Result
        cv2.imshow('Detection Result', frameColor)

        # Define Escape Key
        if escapeWithKeys:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    else:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

video.release()
cv2.destroyAllWindows()