import cv2
import csv
import numpy as np
np.set_printoptions(suppress=True)

# mtx  camera_matrix
mtx = np.loadtxt('cameraMatrix.txt',  delimiter=',')


# dist  distortion_coefficient
dist = np.loadtxt('cameraDistortion.txt',  delimiter=',')


# Read in an image
source = 'source.jpg'
image = cv2.imread(source)


try:
    # Undistort an image
    h,  w = image.shape[:2]
    print("Image to undistort: ", source)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    print("ROI: ", x, y, w, h)

    cv2.imwrite("result.jpg", dst)
    print("Calibrated picture saved as result.jpg")
    print(f"Calibration Matrix: {mtx}")
    print(f"Disortion: {dist}")

except:
    print("an exception occurred")


image = cv2.imread(source)
h,  w = image.shape[:2]
print("Image to undistort: ", source)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (w, h), 5)
dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
print("ROI: ", x, y, w, h)

cv2.imwrite("result.jpg", dst)
print("Calibrated picture saved as calibresult.png")
print(f"Calibration Matrix: {mtx}")
print(f"Disortion: {dist}")
