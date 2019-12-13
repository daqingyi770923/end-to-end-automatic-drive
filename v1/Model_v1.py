import tensorflow as tf
from keras.layers import Conv2D,MaxPool2D,Input,Dense,Dropout,Flatten,Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
import cv2
import random

import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
# config.gpu_options.visible_device_list = "0"
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
KTF.set_session(session)

IMG_H = 100
IMG_W = 100
IMG_C = 1
train_number = 10000

def modelNet():
    img_input = Input(shape=(100,100,1),name='img_input')
    Lam = Lambda(lambda img_input:img_input / 127.5 -1.,input_shape=(100,100,1))(img_input)

    conv_2 = Conv2D(32, (5, 5), padding='same', activation='relu')(Lam)
    pool_2 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(conv_2)

    conv_3 = Conv2D(64, (5, 5), padding='same', activation='relu')(pool_2)
    pool_3 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(conv_3)

    conv_4 = Conv2D(128, (5, 5), padding='same', activation='relu')(pool_3)
    pool_4 = MaxPool2D((3, 3), strides=(2, 2), padding='valid')(conv_4)

    conv_5 = Conv2D(512, (5, 5), padding='same', activation='relu')(pool_4)
    pool_5 = MaxPool2D((3, 3), strides=(3, 3), padding='valid')(conv_5)

    conv_6 = Conv2D(512, (5, 5), padding='same', activation='relu')(pool_5)
    pool_6 = MaxPool2D((3, 3), strides=(3, 3), padding='valid')(conv_6)

    flatten = Flatten()(pool_6)

    dropout = Dropout(0.3)(flatten)

    #prediction steering angle net
    fc_2 = Dense(512,activation='relu')(dropout)
    fc_3 = Dense(256,activation='relu')(fc_2)
    fc_34 = Dense(64,activation='relu')(fc_3)
    prediction_angle = Dense(1,name='output_angle')(fc_34)

    #prediction speed net
    fc_5 = Dense(512,activation='relu')(dropout)
    fc_6 = Dense(256,activation='relu')(fc_5)
    fc_7 = Dense(64,activation='relu')(fc_6)
    prediction_speed = Dense(1,name='output_speed')(fc_7)


    model = Model(inputs = img_input,outputs = [prediction_angle,prediction_speed])
    model.compile(optimizer=Adam(lr=0.0003),loss='mse',loss_weights=[1.,1.])
    filepath = 'Model.h5'
    checkpoint = ModelCheckpoint(filepath,verbose=1,save_weights_only=False,save_best_only=True)
    callback_list = [checkpoint]

    return model,callback_list

def load_dataset(img_path,steering_path,speed_path):
    with open(img_path,'rb') as f:
        imgs = np.array(pickle.load(f))

    with open(steering_path,'rb') as f:
        steerings = np.array(pickle.load(f))

    with open(speed_path,'rb') as f:
        speeds = np.array(pickle.load(f))

    return imgs,steerings,speeds


def augmentData(imgs, steerings,speeds):
    imgs = np.append(imgs, imgs[:, :, ::-1], axis=0)
    steerings = np.append(steerings, -steerings, axis=0)
    speeds = np.append(speeds,speeds,axis=0)
    return imgs, steerings,speeds

def shuffle_Data(imgs,steerings,speeds):
    imgs_shuffle = []
    steerings_shuffle = []
    speeds_shuffle = []
    index = [i for i in range(len(steerings))]
    random.shuffle(index)
    for j in range(len(steerings)):
        imgs_shuffle.append(imgs[index[j]])
        steerings_shuffle.append(steerings[index[j]])
        speeds_shuffle.append(speeds[index[j]])
    imgs_shuffle = np.array(imgs_shuffle)
    steerings_shuffle = np.array(steerings_shuffle)
    speeds_shuffle = np.array(speeds_shuffle)

    return imgs_shuffle,steerings_shuffle,speeds_shuffle

def train():
    imgs,steerings,speeds = load_dataset('train_imgs_100_100_rgb','label_steerings','label_speeds')
    imgs,steerings,speeds = augmentData(imgs,steerings,speeds)
    imgs, steerings, speeds = shuffle_Data(imgs, steerings, speeds)

    train_img = imgs[:train_number,:,:]
    train_steering = steerings[:train_number]
    train_speed = speeds[:train_number]


    test_img = imgs[train_number:,:,:]
    test_steering = steerings[train_number:]
    test_speed = speeds[train_number:]

    train_img = train_img.reshape(train_img.shape[0],100,100,1)
    test_img = test_img.reshape(test_img.shape[0],100,100,1)


    model,callback_list = modelNet()
    model.fit(train_img,[train_steering,train_speed],validation_data=(test_img,[test_steering,test_speed]),
              epochs=60,batch_size=48,callbacks=callback_list)
    model.summary()
    model.save('Model.h5')

train()
