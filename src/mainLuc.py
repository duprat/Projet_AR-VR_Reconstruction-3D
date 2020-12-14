import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import glob
import imutils
from PIL import Image

def display(frame,time,ratio):
    (h, w) = frame.shape[:2]
    resized = cv.resize(frame, (int(w/ratio),int(h/ratio)))
    rotated = imutils.rotate_bound(resized, 90.0)
    cv.imshow('frame',rotated)
    k = cv.waitKey(time) & 0xff
    if k == 27:
        return False
    else:
        return True
    return True

def loadCameraParameters(folder, user, nb_vt_vectors):
    
    matrix = np.genfromtxt(folder + 'camera_matrix_' + user + '.txt',dtype=float)
    dist_coeffs = np.genfromtxt(folder + 'distortion_coeffs_' + user + '.txt',dtype=float)
    
    rotation_vectors = [[] for f in range(nb_vt_vectors)]
    translation_vectors = [[] for f in range(nb_vt_vectors)]
    
    for i in range(nb_vt_vectors):
        rvec = np.genfromtxt(folder + 'rotation_vector' + str(i) + '_' + user + '.txt',dtype=float)
        tvec = np.genfromtxt(folder + 'translation_vector' + str(i) + '_' + user + '.txt',dtype=float)
        
        for r in rvec:
            rotation_vectors[i].append([r])
        for t in tvec:
            translation_vectors[i].append([t])
    
    return matrix, dist_coeffs, rotation_vectors,translation_vectors

nb_images = 14
mtx, dist, rvecs, tvecs = loadCameraParameters('../Text/', 'thomas', nb_images)
## Stereovision
video = cv.VideoCapture('../Videos/blue.mp4')

ret, img1 = video.read()
for i in range(15):
    video.read()
    
ret, img2 = video.read()

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

img1 = cv.undistort(img1, mtx, dist)
img2 = cv.undistort(img2, mtx, dist)

sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)

# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    #img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    #img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
display(img3,0,2)
display(img4,0,2)
display(img5,0,2)
display(img6,0,2)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
#plt.show()

E = cv.transpose(mtx) * F * mtx

s, u, vt = cv.SVDecomp(E)

