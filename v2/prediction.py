# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:35:27 2019

@author: 小蚩尤
"""

import numpy as np
import cv2
import tensorflow as tf
import  time
import pickle
import time

model = tf.keras.models.load_model('drivingMode_v2.h5')

def keras_predict(model, image):
    processed = keras_process_image(image)
    steering_angle,speed = model.predict((processed), batch_size=1)
    steering_angle = float(steering_angle)
    speed = float(speed)
    steering_angle = steering_angle*57.3
    speed = speed*21.479
    return steering_angle,speed


def keras_process_image(img):
    image_x = 300
    image_y = 300
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

f = open('result.txt','w')
def generate_txt(i,steering,speed):
    f.writelines('%d %f %f\n'%(i,steering,speed))

for i in range(1336):
    frame = cv2.imread("dataset//test//%d.tiff"%i)
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (300, 300))

    steering_angle,speed = keras_predict(model, gray)
    generate_txt(i, steering_angle, speed)
    print('steering angle:%f , speed:%f '%(steering_angle,speed))

    steering_angle = -steering_angle
    cv2.imshow('frame', cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
            steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.05)
cv2.destroyAllWindows()
f.close()
