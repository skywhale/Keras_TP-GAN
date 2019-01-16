# -*- coding: utf-8 -*-
"""
This program provides UI to input keypoints of profile image
 and generate frontal image using TP-GAN

Add Keras_TP-GAN directory to PYTHONPATH

"""

import numpy as np
import cv2
import sys
import face_alignment
from keras_tpgan.tpgan import TPGAN

IMG_PATH = './images/test1.jpg'
GENERATOR_WEIGHTS_FILE = './epoch0480_loss0.560.hdf5'

img_size = 128
eye_y = 40/img_size
mouth_y = 88/img_size

EYE_H = int(40/128*img_size); EYE_W = int(40/128*img_size);
NOSE_H = int(32/128*img_size); NOSE_W = int(40/128*img_size);
MOUTH_H = int(32/128*img_size); MOUTH_W = int(48/128*img_size);

def wait():
    while True:
        key = cv2.waitKeyEx(10)
        if key == ord('q'):
            break

tpgan = TPGAN(generator_weights=GENERATOR_WEIGHTS_FILE)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2halfD, device='cpu', flip_input=False)

img = cv2.imread(IMG_PATH)    

landmarks = fa.get_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
assert landmarks is not None

points = landmarks[0]
reye = np.average(np.array((points[37], points[38], points[40], points[41])), axis=0)
leye = np.average(np.array((points[43], points[44], points[46], points[47])), axis=0)
mouth = np.average(np.array((points[51], points[57])), axis=0)
nose_tip = points[30]

print('cropping face')
reye = np.array(reye)
leye = np.array(leye)
nose_tip = np.array(nose_tip)
mouth = np.array(mouth)

vec_mouth2reye = reye - mouth
vec_mouth2leye = leye - mouth
# angle reye2mouth against leye2mouth
phi = np.arccos(vec_mouth2reye.dot(vec_mouth2leye) / (np.linalg.norm(vec_mouth2reye) * np.linalg.norm(vec_mouth2leye)))/np.pi * 180

if phi < 15: # consider the profile image is 90 deg.
    # in case of 90 deg. set invisible eye with copy of visible eye.
    eye_center = (reye + leye) / 2
    if nose_tip[0] > eye_center[0]:
        leye = reye
    else:
        reye = leye

# calc angle eyes against horizontal as theta
if np.array_equal(reye, leye) or phi < 38: # in case of 90 deg. avoid rotation
    theta = 0
else: 
    vec_leye2reye = reye - leye
    if vec_leye2reye[0] < 0:
        vec_leye2reye = -vec_leye2reye
    theta = np.arctan(vec_leye2reye[1]/vec_leye2reye[0])/np.pi*180

imgcenter = (img.shape[1]/2, img.shape[0]/2)
rotmat = cv2.getRotationMatrix2D(imgcenter, theta, 1)
rot_img = cv2.warpAffine(img, rotmat, (img.shape[1], img.shape[0])) 

crop_size = int((mouth[1] - reye[1])/(mouth_y - eye_y))

crop_up = int(reye[1] - crop_size * eye_y)
if crop_up < 0:
    crop_up = 0
    
crop_down = crop_up + crop_size
if crop_down > rot_img.shape[0]:
    crop_down = rot_img.shape[0]
    
crop_left = int((reye[0] + leye[0]) / 2 - crop_size / 2)
if crop_left < 0:
    crop_left = 0
    
crop_right = crop_left + crop_size
if crop_right > rot_img.shape[1]:
    crop_right = rot_img.shape[1]

crop_img = rot_img[crop_up:crop_down, crop_left:crop_right]
crop_img = cv2.resize(crop_img, (img_size, img_size))

landmarks = fa.get_landmarks(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
assert landmarks is not None
points = landmarks[0]

print('cropping parts')
leye_points = points[42:48]
leye_center = (np.max(leye_points, axis=0) + np.min(leye_points, axis=0)) / 2
leye_left = int(leye_center[0] - EYE_W / 2)
leye_up = int(leye_center[1] - EYE_H / 2)
leye_img = crop_img[leye_up:leye_up + EYE_H, leye_left:leye_left + EYE_W].copy()

reye_points = points[36:42]
reye_center = (np.max(reye_points, axis=0) + np.min(reye_points, axis=0)) / 2
reye_left = int(reye_center[0] - EYE_W / 2)
reye_up = int(reye_center[1] - EYE_H / 2)
reye_img = crop_img[reye_up:reye_up + EYE_H, reye_left:reye_left + EYE_W].copy()

nose_points = points[31:36]
nose_center = (np.max(nose_points, axis=0) + np.min(nose_points, axis=0)) / 2
nose_left = int(nose_center[0] - NOSE_W / 2)
nose_up = int(nose_center[1] - 10 - NOSE_H / 2)
nose_img = crop_img[nose_up:nose_up + NOSE_H, nose_left:nose_left + NOSE_W].copy()

mouth_points = points[48:60]
mouth_center = (np.max(mouth_points, axis=0) + np.min(mouth_points, axis=0)) / 2
mouth_left = int(mouth_center[0] - MOUTH_W / 2)
mouth_up = int(mouth_center[1] - MOUTH_H / 2)
mouth_img = crop_img[mouth_up:mouth_up + MOUTH_H, mouth_left:mouth_left + MOUTH_W].copy()

show_img = np.copy(crop_img)
for (x, y) in [reye_center, leye_center, nose_center, mouth_center]:
    cv2.circle(show_img, (x, y), 3, (255, 0, 0), -1)
cv2.imshow("orig_img", show_img)
wait()

x_z = np.random.normal(scale=0.02, size=(1, 100))
x_face = (cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
x_leye = (cv2.cvtColor(leye_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
x_reye = (cv2.cvtColor(reye_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
x_nose = (cv2.cvtColor(nose_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]
x_mouth = (cv2.cvtColor(mouth_img, cv2.COLOR_BGR2RGB).astype(np.float)/255)[np.newaxis,:]

[pred_faces, pred_faces64, pred_faces32, pred_leyes, pred_reyes, pred_noses, pred_mouthes
        ] = tpgan.generate([x_face, x_leye, x_reye, x_nose, x_mouth, x_z])
cv2.imshow("pred_img", pred_faces[0])
wait()

cv2.destroyAllWindows()   
