import os
import cv2 as cv
import numpy as np
import imutils

def isValid(filepath):
    return os.path.exists(filepath) and os.path.isfile(filepath)

def display(frame):
    (h, w) = frame.shape[:2]
    resized = cv.resize(frame, (int(w/2),int(h/2)))
    rotated = imutils.rotate_bound(resized, 90.0)
    cv.imshow('frame',rotated)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        return False
    else:
        return True
    return True

def main(filepath):
    video = cv.VideoCapture(filepath)
    
    nb_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    
    for i in range(nb_frames -1):
        ret,frame = video.read()
        
        
        if not display(frame): 
            break
    cv.destroyAllWindows()


filepath = "../Videos/blue.mp4"

if isValid(filepath):
    
    main(filepath)

else:
    print("file not valid")