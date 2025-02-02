import os

import cv2 as cv
import imutils
import matplotlib.pylab as plt
import numpy as np


def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)


def is_BGR(img):
    if img.shape[2] == 3:
        return True
    else:
        return False


def display(frame, time, ratio, title='Image'):
    (h, w) = frame.shape[:2]
    resized = cv.resize(frame, (int(w / ratio), int(h / ratio)))
    rotated = imutils.rotate_bound(resized, 90.0)
    cv.imshow(title, rotated)
    k = cv.waitKey(time) & 0xff
    if k == 27:
        return False
    else:
        return True


def displayMultiple(images, time, ratio):
    accu = 0
    for image in images:
        (h, w) = image.shape[:2]
        resized = cv.resize(image, (int(w / ratio), int(h / ratio)))
        rotated = imutils.rotate_bound(resized, 90.0)
        cv.imshow(str(accu), rotated)
        accu += 1
    k = cv.waitKey(time) & 0xff
    if k == 27:
        cv.destroyAllWindows()


def loadSequence(filename, _format, start, end):
    images = list()

    for i in range(start, end):
        imgName = filename + str(i) + _format
        if isValid(imgName):
            images.append(cv.imread(imgName, cv.IMREAD_UNCHANGED))
        else:
            print("Error reading image ", i, ": it doesn't exist.")

    return images


def buildDisparityMap(imgL, imgR):
    imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL, imgR)
    display(disparity, 0, 4, title='Disparity Map')


def loadCameraParameters(user, nb_vt_vectors):
    matrix = np.genfromtxt(user + 'Text/camera_matrix.txt', dtype=float)
    dist_coeffs = np.genfromtxt(user + 'Text/distortion_coeffs.txt', dtype=float)

    rotation_vectors = [[] for f in range(nb_vt_vectors)]
    translation_vectors = [[] for f in range(nb_vt_vectors)]

    for i in range(nb_vt_vectors):
        rvec = np.genfromtxt(user + 'Text/rotation_vector_' + str(i) + '.txt', dtype=float)
        tvec = np.genfromtxt(user + 'Text/translation_vector_' + str(i) + '.txt', dtype=float)

        for r in rvec:
            rotation_vectors[i].append([r])
        for t in tvec:
            translation_vectors[i].append([t])

    return matrix, dist_coeffs, rotation_vectors, translation_vectors


def saveCameraParameters(user, mtx, dist, rvecs, tvecs):
    np.savetxt(user + "Text/camera_matrix.txt", mtx)
    np.savetxt(user + "Text/distortion_coeffs.txt", dist)

    accu = 0
    for vec in rvecs:
        np.savetxt(user + "Text/rotation_vector_" + str(accu) + ".txt", vec)
        accu += 1

    accu = 0
    for vec in tvecs:
        np.savetxt(user + "Text/translation_vector_" + str(accu) + ".txt", vec)
        accu += 1


def siftDetector(image):
    sift = cv.SIFT_create()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    key_points, descriptors = sift.detectAndCompute(gray, None)
    return key_points, descriptors


def computeReProjectionError(mtx, dist, rvecs, tvecs, objpoints, imgpoints):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))


def getFramesWithCorners(user, CHESSBOARD):
    calibration_path = user + "Videos/Calib_2.mp4"
    calibration_video = cv.VideoCapture(calibration_path)

    nb_frames = int(calibration_video.get(cv.CAP_PROP_FRAME_COUNT))

    accu = 0
    for i in range(0, nb_frames - 1):
        icht, frame = calibration_video.read()

        if icht:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, CHESSBOARD,
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                cv.imwrite(user + "Images/calib_" + str(accu) + ".jpg", frame)
                accu += 1
    return accu


def undistort(image, mtx, dist):
    h, w = image.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort

    # option 1
    dst = cv.undistort(image, mtx, dist, None, newcameramtx)

    # option 2
    # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # dst = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    return dst


def computeCameraParameters(user, images):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for image in images:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    for image in images:
        image = undistort(image, mtx, dist)

    computeReProjectionError(mtx, dist, rvecs, tvecs, objpoints, imgpoints)

    saveCameraParameters(user, mtx, dist, rvecs, tvecs)

    return mtx, dist, rvecs, tvecs


def drawlines(img1, img2, lines, pts1, pts2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    if not is_BGR(img1):
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    if not is_BGR(img2):
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, 7)
        img2 = cv.circle(img2, tuple(pt2), 5, color, 7)
    return img1, img2


def compute3DPoints(P1, P2, pts1, pts2):
    # print('1: ', pts1[0])
    # print('2: ', pts2[0])
    p1_3D = np.ones((2, pts1.shape[0]))
    p2_3D = np.ones((2, pts2.shape[0]))
    p1_3D[0], p1_3D[1] = pts1[:, 0].copy(), pts1[:, 1].copy()
    p2_3D[0], p2_3D[1] = pts2[:, 0].copy(), pts2[:, 1].copy()

    pts_obtained = cv.triangulatePoints(P1[:3], P2[:3], p1_3D, p1_3D)

    good_pts_mask = np.where(pts_obtained[3] != 0)[0]
    point_4d = pts_obtained / pts_obtained[3]
    points_3d = point_4d[:3, :].T

    return points_3d, good_pts_mask
    """
    pts3d /= pts3d[3]
    X1 = P1[:3] @ pts3d
    X2 = P2[:3] @ pts3d
    return pts3d[:3], X1, X2 
    """


def computeEpilines(img1, img2, camera_matrix):
    kp1, des1 = siftDetector(img1)
    kp2, des2 = siftDetector(img2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=500)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    E, mask = cv.findEssentialMat(pts1, pts2, camera_matrix)
    # Eprime, mask = cv.findEssentialMat(pts2,pts1,camera_matrix)

    # print("E:\n",E)
    # print("Eprime:\n",Eprime)
    # print(E == Eprime)

    pts3, R, t, mask = cv.recoverPose(E, pts1, pts2, camera_matrix)
    # pts4, R2, t2, mask = cv.recoverPose(Eprime,pts2,pts1,camera_matrix)

    Projection_matrix = np.hstack((R, t))
    # Projection_matrix2 = np.hstack((R2, t2))

    # print('pts3\n',pts3)
    # print('pts4\n',pts4)
    # print(Projection_matrix == Projection_matrix2)
    # print('Projection_matrix\n',Projection_matrix)
    # print('Projection_matrix2\n',Projection_matrix2)

    pm1 = np.eye(3, 4)

    # points3Dworld, frame1C,frame2C = compute3DPoints(Projection_matrix,Projection_matrix2,pts1,pts2)
    # points3Dworld, goodPointsMask = compute3DPoints(pm1,Projection_matrix,pts1,pts2)
    points3Dworld, goodPointsMask = compute3DPoints(pm1, camera_matrix @ Projection_matrix, pts1, pts2)

    print(points3Dworld)

    """
    ax = plt.axes(projection='3d')
    ax.scatter(points3Dworld[:, 0], points3Dworld[:, 1], points3Dworld[:, 2], s=1)
    plt.show()

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    displayMultiple([img3, img5], 0, 4)
    """
    return points3Dworld


def main(user, CHESSBOARD):
    # object_path = user + "Videos/blue.mp4"

    # nb_images = getFramesWithCorners(user, CHESSBOARD)

    nb_images = 14

    # images = loadSequence(user + 'Images/Calibration/calib_', '.jpg',0,nb_images)

    # mtx, dist, rvecs, tvecs = computeCameraParameters(user,images) # 0.1120613588987386

    # poseEstimation(user, nb_images, images[0], images[1])

    mtx, dist, rvecs, tvecs = loadCameraParameters(user, nb_images)

    video = cv.VideoCapture(user + "Videos/blue.mp4")
    length = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    step = 3

    pointCloud = list()

    for i in range(length - 1):
        ret, imgL = video.read()
        for j in range(step):
            if (i+j) < (length - 1):
                video.read()
        ret, imgR = video.read()

        TimgL = undistort(imgL, mtx, dist)
        TimgR = undistort(imgR, mtx, dist)

        _3Dpoints = computeEpilines(imgL, imgR, mtx)



        ax = plt.axes(projection='3d')
        ax.scatter(pointCloud[:, 0], pointCloud[:, 1], pointCloud[:, 2], s=1)
        plt.show()
    """
    imgL = cv.imread(user + "Images/Blue/L.jpg", cv.IMREAD_UNCHANGED)
    imgR = cv.imread(user + "Images/Blue/R.jpg", cv.IMREAD_UNCHANGED)

    TimgL = undistort(imgL, mtx, dist)
    TimgR = undistort(imgR, mtx, dist)

    _3Dpoints = computeEpilines(imgL, imgR, mtx)
    """



user = "../Thomas/"

CHESSBOARD = (7, 7)
main(user, CHESSBOARD)
