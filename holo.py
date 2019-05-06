import cv2
import numpy as np
import os,sys

def holo(original,scale=0.5,scaleR=4,distance=0):
    '''
        Create 3D hologram from image (must have equal dimensions)
    '''
    
    height = int((scale*original.shape[0]))
    width = int((scale*original.shape[1]))
    
    image = cv2.resize(original, (width, height), interpolation = cv2.INTER_CUBIC)
    
    up = image.copy()
    down = rotate_bound(image.copy(),180)
    right = rotate_bound(image.copy(), 90)
    left = rotate_bound(image.copy(), 270)
    
    hologram = np.zeros([max(image.shape)*scaleR+distance,max(image.shape)*scaleR+distance,3], image.dtype)
    
    center_x = (hologram.shape[0])/2
    center_y = (hologram.shape[1])/2
    
    vert_x = (up.shape[0])/2
    vert_y = (up.shape[1])/2
    hologram[0:up.shape[0], center_x-vert_x+distance:center_x+vert_x+distance] = up
    hologram[ hologram.shape[1]-down.shape[1]:hologram.shape[1] , center_x-vert_x+distance:center_x+vert_x+distance] = down
    hori_x = (right.shape[0])/2
    hori_y = (right.shape[1])/2
    hologram[ center_x-hori_x : center_x-hori_x+right.shape[0] , hologram.shape[1]-right.shape[0]+distance : hologram.shape[1]+distance] = right
    hologram[ center_x-hori_x : center_x-hori_x+left.shape[0] , 0+distance : left.shape[0]+distance ] = left
    
    cv2.imwrite("/home/gopika/Project/Sketch-Photo-Conversion-using-Deep-CNN-master/Main Project/static/img/hologram.jpg",hologram)

  


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
    
if __name__ == '__main__' :
    orig = cv2.imread('/home/gopika/Project/Sketch-Photo-Conversion-using-Deep-CNN-master/Main Project/static/img/output/output.jpg')
    holo(orig,scale=1.0)

