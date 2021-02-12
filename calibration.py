"""
From https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration

Calling:
cameracalib.py  <folder> <image type> <num rows> <num cols> <cell dimension>

like cameracalib.py folder_name png

--h for help
"""

import argparse
import sys
import glob
import cv2
import numpy as np
np.set_printoptions(suppress=True)

# ---------------------- SET THE PARAMETERS
nRows = 9
nCols = 6
dimension = 25  # - mm

workingFolder = "."
imageType = 'jpg'
# ------------------------------------------

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
obj_point = np.zeros((nRows*nCols, 3), np.float32)
obj_point[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []  # 3d point in real world space
img_points = []  # 2d points in image plane.

if len(sys.argv) < 6:
    print("\n Not enough inputs are provided. Using the default values.\n\n"
          " type -h for help")
else:
    workingFolder = sys.argv[1]
    imageType = sys.argv[2]
    nRows = int(sys.argv[3])
    nCols = int(sys.argv[4])
    dimension = float(sys.argv[5])

if '-h' in sys.argv or '--h' in sys.argv:
    print("\n IMAGE CALIBRATION GIVEN A SET OF IMAGES")
    print(" call: python calibration.py <folder> <image type> <num rows (9)> <num cols (6)> <cell dimension (25)>")
    print("\n The script will look for every image in the provided folder and will show the pattern found."
          " User can skip the image pressing ESC or accepting the image with RETURN. "
          " At the end the end the following files are created:"
          "  - cameraDistortion.txt"
          "  - cameraMatrix.txt \n\n")

    sys.exit()

# Find the images files
filename = workingFolder + "/*." + imageType
images = glob.glob(filename)

print(len(images))
if len(images) < 9:
    print("Not enough images were found: at least 9 shall be provided!")
    sys.exit()


else:
    nPatternFound = 0
    imgNotGood = images[1]

    for fname in images:
        if 'calibresult' in fname:
            continue
        # -- Read the file and convert in greyscale
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(f"Reading image: {fname}")

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Pattern found! Press ESC to skip or ENTER to accept")
            # --- Sometimes, Harris cornes fails with crappy pictures, so
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nCols, nRows), corners2, ret)
            cv2.imshow('img', img)
            # cv2.waitKey(0)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:  # -- ESC Button
                print("Image Skipped")
                imgNotGood = fname
                continue

            print("Image accepted")
            nPatternFound += 1
            obj_points.append(obj_point)
            img_points.append(corners2)

            # cv2.waitKey(0)
        else:
            imgNotGood = fname


cv2.destroyAllWindows()

if (nPatternFound > 1):
    print(f"Found {nPatternFound} valid images ")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)

    # Undistort an image
    img = cv2.imread(imgNotGood)
    h,  w = img.shape[:2]
    print("Image to undistort: ", imgNotGood)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    print("ROI: ", x, y, w, h)

    cv2.imwrite("calibresult.jpg", dst)
    print("Calibrated picture saved as result.jpg")
    print(f"Calibration Matrix: {mtx}")
    print(f"Disortion: {dist}")

    # --------- Save result
    filename = "cameraMatrix.txt"
    np.savetxt(filename, mtx, delimiter=',')
    filename = "cameraDistortion.txt"
    np.savetxt(filename, dist, delimiter=',')

    # Re-projection error gives a good estimation of just how exact the found parameters are.
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points2,
                         cv2.NORM_L2)/len(img_points2)
        mean_error += error

    print("total error: ", mean_error/len(obj_points))

else:
    print("In order to calibrate you need at least 9 good pictures... try again")
