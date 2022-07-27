import os

# GPU無効化
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# バージョン?
# os.environ["SM_FRAMEWORK"] = "tf.keras"

# import tensorflow.python.keras as keras
# from tensorflow.python.keras import layers
# from tensorflow.python.keras import models
# from tensorflow.keras import applications
# from tensorflow.keras import optimizers
# from tensorflow.python.keras import callbacks
# from tensorflow.python.keras import metrics
# from tensorflow.python.keras.datasets import cifar10
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
# import tensorflow.python.keras.backend as K

import tensorflow.python.keras as keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import applications
from tensorflow.keras import optimizers
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
# import tensorflow.python.keras.backend as K

# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import signal
from pathlib import Path
import glob
# import cv2
import pickle
import re
import time
import datetime
import os
import random
import math
import copy
from random import shuffle
from itertools import islice,chain
from pprint import pprint
import PIL
import json

from tensorflow.python.util.nest import _yield_value

from defined_models import efficientnetv2
from originalnet import *
from ImageIterator import *
from ImageSequenceIterator import *


### GPU稼働確認 ###
import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)
print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

strategy = tf.distribute.MirroredStrategy()
    

### ROC AUC ###
# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc

# def roc_auc(y_true, y_pred):
#     roc_auc = tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
#     return roc_auc

# def roc(y_true, y_pred):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#     return fpr, tpr, thresholds


###サンプルデータ取得###
def getSampleData():
    # CIFAR10データの読み込み
    (X_train, y_train), (X_test, y_test), = cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # 0番目のクラスと1番目のクラスのデータを結合
    X_train = np.concatenate([X_train[y_train == 0], X_train[y_train == 1]], axis=0)
    y_train = np.concatenate([y_train[y_train == 0], y_train[y_train == 1]], axis=0)
    X_test = np.concatenate([X_test[y_test == 0], X_test[y_test == 1]], axis=0)
    y_test = np.concatenate([y_test[y_test == 0], y_test[y_test == 1]], axis=0)
    sep = math.floor(len(X_test)/2)
    X_val = X_test[:sep]
    y_val = y_test[:sep]
    X_test = X_test[sep:]
    y_test = y_test[sep:]

    # Generator生成
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
    test_generator = test_datagen.flow(X_test, y_test, batch_size=32)

    return (train_generator,val_generator,test_generator)


###評価関数取得###
def getMetrics(mode=None):
    mode = "all"
    metrics_list = []
    metrics_list.append('accuracy')
    # metrics_list.append(metrics.Accuracy())
    if mode!="accuracy":
        metrics_list.append(metrics.AUC())
    if mode=="all":
        metrics_list.append(metrics.Precision())
        metrics_list.append(metrics.Recall())
        # metrics_list.append(metrics.SpecificityAtSensitivity())
        # metrics_list.append(metrics.metrics.SensitivityAtSpecificity())
    if mode!="accuracy":
        metrics_list.append(metrics.TruePositives())
        metrics_list.append(metrics.TrueNegatives())
        metrics_list.append(metrics.FalsePositives())
        metrics_list.append(metrics.FalseNegatives())
    return metrics_list








### CNN 2値分類 ###
def loadSampleCnn(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="SampleCNN")
        model.add(layers.Conv2D(16, (3, 3), activation='relu', data_format='channels_last', input_shape=input_shape))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        # model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        # model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(1, activation='sigmoid'))

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics()
        )
    return model


### CNN VGG16 ###
def loadVgg16(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        # input_tensor = models.Input(shape=input_shape)
        # vgg16 = applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=input_tensor)
        # _model = models.Sequential()
        # _model.add(layers.Flatten(input_shape=vgg16.output_shape[1:]))
        # _model.add(layers.Dense(256, activation='relu'))
        # _model.add(layers.Dropout(0.5))
        # _model.add(layers.Dense(1, activation='softmax'))
        # model = models.Model(inputs=vgg16.input, outputs=_model(vgg16.output), name="VGG16")
        # # for layer in model.layers[:15]:
        # #     layer.trainable = False

        model = models.Sequential(name="VGG16")
        model.add(layers.Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=1, activation="sigmoid"))

        # model = models.Sequential(name="VGG16")
        # model.add(applications.vgg16.VGG16(include_top=False, weights=None, input_shape=input_shape))
        # model.add(layers.Flatten())
        # model.add(layers.Dense(256, activation='relu'))
        # model.add(layers.Dropout(0.5))
        # model.add(layers.Dense(1, activation='softmax'))

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model

### CNN Inception ###
def loadInceptionV3(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="InceptionV3")
        model.add(applications.inception_v3.InceptionV3(include_top=False, weights=None, input_shape=input_shape))
        print(len(applications.inception_v3.InceptionV3(include_top=False, weights=None, input_shape=input_shape).layers))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model

### CNN Xception ###
def loadXception(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="Xception")
        model.add(applications.xception.Xception(include_top=False, weights=None, input_shape=input_shape))
        # print(len(applications.xception.Xception(include_top=False, weights=None, input_shape=input_shape).layers))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            optimizer="sgd",
            metrics=getMetrics("all")
        )
    return model

### CNN Xception (素) ###
def loadXceptionOriginal(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        inputs = keras.Input(shape=input_shape)

        # entry flow
        x = layers.Convolution2D(32, (3,3), strides=2)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Convolution2D(64, (3,3))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        residual = layers.Convolution2D(128, (1,1), strides=2, padding='same')(x)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        residual = layers.Convolution2D(256, (1,1), strides=2, padding='same')(x)
        residual = layers.BatchNormalization()(residual)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        residual = layers.Convolution2D(728, (1,1), strides=2, padding='same')(x)
        residual = layers.BatchNormalization()(residual)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])

        # middle flow
        for i in range(8):
            residual = x
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(728, (3,3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(728, (3,3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv2D(728, (3,3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.add([x, residual])
        
        # exit flow
        residual = layers.Convolution2D(1024, (1,1), strides=2, padding='same')(x)
        residual = layers.BatchNormalization()(residual)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(728, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(1024, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.SeparableConv2D(1536, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(2048, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # x = layers.GlobalAveragePooling2D()(x)
        # x = layers.Dense(1, kernel_initializer='he_normal', activation='sigmoid')(x)
    
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x)

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model

### CNN EfficientNetV2 ###
def loadEfficientNetV2(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="EfficientNetV2")
        model.add(layers.InputLayer(input_shape=input_shape))
        # model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-b0', include_top=False))
        model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-b3', include_top=False))
        model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(1, activation='sigmoid'))

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model



### オリジナル構造 ###
def original_net(inputs,output_filter=128):
    cell_num = 8
    x = inputs

    # cell1 = layers.Conv2D(output_filter//cell_num,(1,1),padding="same")(x)
    # cell2 = layers.AveragePooling2D(pool_size=(2,2),strides=1,padding="same")(x)
    # cell2 = layers.Conv2D(output_filter//cell_num,(3,3),padding="same")(cell2)
    # cell3 = layers.Conv2D(output_filter//cell_num,(1,1),padding="same")(x)
    # cell3 = layers.Conv2D(output_filter//cell_num,(3,3),padding="same")(cell3)
    # cell4 = layers.Conv2D(output_filter//cell_num,(1,1),padding="same")(x)
    # cell4 = layers.Conv2D(output_filter//cell_num,(3,3),padding="same")(cell4)
    # cell4 = layers.Conv2D(output_filter//cell_num,(5,5),padding="same")(cell4)
    # cell5_residual = layers.Conv2D(output_filter//2//cell_num,(1,1),padding="same")(x)
    # cell5 = layers.Conv2D(output_filter//2//cell_num,(1,1),padding="same")(x)
    # cell5 = layers.Concatenate()([cell5_residual,cell5])
    # cell5_residual = layers.Conv2D(output_filter//2//cell_num,(1,1),padding="same")(cell5)
    # cell5 = layers.Conv2D(output_filter//2//cell_num,(3,3),padding="same")(cell5)
    # cell5 = layers.Concatenate()([cell5_residual,cell5])
    # cell5_residual = layers.Conv2D(output_filter//2//cell_num,(1,1),padding="same")(cell5)
    # cell5 = layers.Conv2D(output_filter//2//cell_num,(5,5),padding="same")(cell5)
    # cell5 = layers.Concatenate()([cell5_residual,cell5])
    # cell6 = layers.Activation('relu')(x)
    # cell6 = layers.SeparableConv2D(output_filter//cell_num, (3,3), padding='same')(cell6)
    # cell7 = layers.Activation('relu')(x)
    # cell7 = layers.SeparableConv2D(output_filter//cell_num, (3,3), padding='same')(cell7)
    # cell8 = layers.Activation('relu')(x)
    # cell8 = layers.SeparableConv2D(output_filter//cell_num, (3,3), padding='same')(cell8)
    # x = layers.Concatenate()([cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8])

    red = layers.SeparableConv2D(output_filter//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(x)
    cell1 = red
    cell2 = layers.AveragePooling2D(pool_size=(2,2),strides=1,padding="same")(x)
    cell2 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell2)
    cell3 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
    cell4 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
    cell5 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
    cell5 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell5)
    cell6 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
    cell6 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell6)
    cell7 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
    cell7 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell7)
    cell7 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell7)
    cell8_residual = layers.SeparableConv2D(output_filter//2//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(red)
    cell8 = layers.SeparableConv2D(output_filter//2//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
    cell8 = layers.Concatenate()([cell8_residual,cell8])
    cell8_residual = layers.SeparableConv2D(output_filter//2//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(cell8)
    cell8 = layers.SeparableConv2D(output_filter//2//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell8)
    cell8 = layers.Concatenate()([cell8_residual,cell8])
    cell8_residual = layers.SeparableConv2D(output_filter//2//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(cell8)
    cell8 = layers.SeparableConv2D(output_filter//2//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell8)
    cell8 = layers.Concatenate()([cell8_residual,cell8])
    x = layers.Concatenate()([cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8])

    # x = layers.DepthwiseConv2D((3,3), strides=2, padding='same')(x)

    # cp = layers.Conv2D(output_filter//2, (1,1), strides=2, padding="same")(x)
    # x = layers.SeparableConv2D(output_filter//2, (3,3), strides=2, padding='same')(x)
    # x = layers.Concatenate()([x,cp])
    # x = layers.BatchNormalization()(x)
    return x

# def original_net2(inputs,output_filter=128):
#     cell_num = 8
#     x = inputs
#     cell1 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell2 = layers.MaxPool2D(pool_size=(2,2),strides=1,padding="same")(x)
#     cell2 = layers.Conv2D(output_filter//cell_num,1,padding="same")(cell2)
#     cell3 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell3 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell3)
#     cell4 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell4 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell4)
#     cell4 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell4)
#     cell5 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell5 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell5)
#     cell5 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell5)
#     cell5 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell5)
#     cell6 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell6 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell6)
#     cell6 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell6)
#     cell6 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell6)
#     cell6 = layers.Conv2D(output_filter//cell_num,9,padding="same")(cell6)
#     cell7 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell7 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell7)
#     cell7 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell7)
#     cell7 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell7)
#     cell7 = layers.Conv2D(output_filter//cell_num,9,padding="same")(cell7)
#     cell7 = layers.Conv2D(output_filter//cell_num,11,padding="same")(cell7)
#     cell8 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
#     cell8 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell8)
#     cell8 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell8)
#     cell8 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell8)
#     cell8 = layers.Conv2D(output_filter//cell_num,9,padding="same")(cell8)
#     cell8 = layers.Conv2D(output_filter//cell_num,11,padding="same")(cell8)
#     cell8 = layers.Conv2D(output_filter//cell_num,13,padding="same")(cell8)
#     x = layers.Concatenate()([cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8])

#     x = layers.SeparableConv2D(output_filter, (3,3), strides=2, padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     return x

def loadOriginalNet(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        inputs = keras.Input(shape=input_shape)
        x = inputs

        ###1つ目###
        x = layers.Conv2D(32, (3,3), strides=2, padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # (128,128,64)

        residual = x
        residual = layers.Conv2D(128, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(128, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = layers.Conv2D(256, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(256, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(256, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = layers.Conv2D(512, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        # (16,16,512)

        x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        for i in range(5):
            residual = x
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = original_net(x,512)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = original_net(x,512)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(rate=0.3)(x)
            x = original_net(x,512)
            x = layers.add([x, residual])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # (16,16,512)


        residual = x
        residual = layers.Conv2D(768, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(768, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(768, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = x
        x = original_net(x,768)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = original_net(x,768)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        # (8,8,768)

        residual = x
        residual = layers.Conv2D(1024, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(1024, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(1024, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        # (4,4,1024)

        x = layers.SeparableConv2D(1536, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(2048, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(rate=0.5)(x)
        # (4,4,2048)



        ###2つ目###
        # x = layers.SeparableConv2D(32, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)

        # residual = layers.SeparableConv2D(64, (3,3), strides=2, padding="same")(x)
        # residual = layers.BatchNormalization()(residual)
        # x = original_net(x,64)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # x = original_net(x,64)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # # x = original_net(x,64)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(64, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.add([x,residual])
        # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(64, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.Activation('relu')(x)
        # # ()

        # # residual = x
        # # for i in range(4):
        # #     x = layers.Activation('relu')(x)
        # #     x = layers.SeparableConv2D(64, (3,3), padding='same')(x)
        # #     x = layers.BatchNormalization()(x)
        # #     x = layers.Activation('relu')(x)
        # #     x = layers.SeparableConv2D(64, (3,3), padding='same')(x)
        # #     x = layers.BatchNormalization()(x)
        # #     x = layers.Activation('relu')(x)
        # #     x = layers.SeparableConv2D(64, (3,3), padding='same')(x)
        # #     x = layers.BatchNormalization()(x)
        # #     x = layers.add([x, residual])
        # # x = layers.Activation('relu')(x)

        # residual = layers.SeparableConv2D(128, (3,3), strides=2, padding="same")(x)
        # residual = layers.BatchNormalization()(residual)
        # x = original_net(x,128)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # x = original_net(x,128)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # # x = original_net(x,128)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(128, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.add([x,residual])
        # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(128, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.Activation('relu')(x)

        # # residual = x
        # # for i in range(4):
        # #     x = layers.Activation('relu')(x)
        # #     x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
        # #     x = layers.BatchNormalization()(x)
        # #     x = layers.Activation('relu')(x)
        # #     x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
        # #     x = layers.BatchNormalization()(x)
        # #     # x = layers.Activation('relu')(x)
        # #     # x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
        # #     # x = layers.BatchNormalization()(x)
        # #     x = layers.add([x, residual])
        # # x = layers.Activation('relu')(x)

        # residual = layers.SeparableConv2D(256, (3,3), strides=2, padding="same")(x)
        # residual = layers.BatchNormalization()(residual)
        # x = original_net(x,256)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # x = original_net(x,256)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # # x = original_net(x,256)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(256, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.add([x,residual])
        # # (32,32,256)

        # for i in range(4):
        #     residual = x
        #     x = layers.Activation('relu')(x)
        #     x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
        #     x = layers.BatchNormalization()(x)
        #     x = layers.Activation('relu')(x)
        #     x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
        #     x = layers.BatchNormalization()(x)
        #     # x = layers.Activation('relu')(x)
        #     # x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
        #     # x = layers.BatchNormalization()(x)
        #     x = layers.add([x, residual])
        # x = layers.Activation('relu')(x)
        # # (32,32,256)

        # residual = layers.SeparableConv2D(512, (3,3), strides=2, padding="same")(x)
        # residual = layers.BatchNormalization()(residual)
        # x = original_net(x,512)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # x = original_net(x,512)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # # x = original_net(x,512)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(512, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.add([x,residual])
        # x = layers.Activation('relu')(x)

        # residual = layers.SeparableConv2D(768, (3,3), strides=2, padding="same")(x)
        # residual = layers.BatchNormalization()(residual)
        # x = original_net(x,768)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # x = original_net(x,768)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # # x = original_net(x,768)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(768, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        # x = layers.add([x,residual])
        # # (8,8,768)

        # for i in range(4):
        #     residual = x
        #     x = layers.Activation('relu')(x)
        #     x = layers.SeparableConv2D(768, (3,3), padding='same')(x)
        #     x = layers.BatchNormalization()(x)
        #     x = layers.Activation('relu')(x)
        #     x = layers.SeparableConv2D(768, (3,3), padding='same')(x)
        #     x = layers.BatchNormalization()(x)
        #     # x = layers.Activation('relu')(x)
        #     # x = layers.SeparableConv2D(768, (3,3), padding='same')(x)
        #     # x = layers.BatchNormalization()(x)
        #     x = layers.add([x, residual])
        # x = layers.Activation('relu')(x)
        # # (8,8,768)

        # # residual = layers.SeparableConv2D(768, (1,1), strides=2, padding="same")(x)
        # # residual = layers.BatchNormalization()(residual)
        # # x = original_net(x,768)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # # x = original_net(x,768)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # # x = original_net(x,768)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.Activation('relu')(x)
        # # x = layers.SeparableConv2D(768, (3,3), strides=2, padding="same")(x)
        # # x = layers.BatchNormalization()(x)
        # # x = layers.add([x,residual])
        # # x = layers.Activation('relu')(x)

        # x = layers.Dropout(rate=0.4)(x)
        # x = layers.SeparableConv2D(1024, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)
        # x = layers.SeparableConv2D(2048, (3,3), padding="same")(x)
        # x = layers.BatchNormalization()(x)
        # x = layers.Activation('relu')(x)



        x = layers.GlobalAveragePooling2D()(x)
        # x = layers.Flatten()(x)
        x = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x, name="OriginalNet")

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model


def loadOriginalNetNonDrop(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        inputs = keras.Input(shape=input_shape)
        x = inputs

        ###1つ目###
        x = layers.Conv2D(32, (3,3), strides=2, padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # (128,128,64)

        residual = x
        residual = layers.Conv2D(128, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(128, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = layers.Conv2D(256, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(256, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(256, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = layers.Conv2D(512, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        # (16,16,512)

        x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        for i in range(5):
            residual = x
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = original_net(x,512)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = original_net(x,512)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            # x = layers.Dropout(rate=0.3)(x)
            x = original_net(x,512)
            x = layers.add([x, residual])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # (16,16,512)


        residual = x
        residual = layers.Conv2D(768, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(768, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(768, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        residual = x
        x = original_net(x,768)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = original_net(x,768)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        # (8,8,768)

        residual = x
        residual = layers.Conv2D(1024, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(1024, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(1024, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        # (4,4,1024)

        x = layers.SeparableConv2D(1536, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(2048, (3,3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # x = layers.Dropout(rate=0.5)(x)
        # (4,4,2048)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x, name="OriginalNetNonDrop")

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model









### CNN AutoEncoder ###
def loadAutoEncoder(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        input_img = models.Input(shape=input_shape)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_img)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        # x = layers.ZeroPadding2D(padding=((2, 2), (0, 0)), data_format=None)(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        y = layers.Conv2D(3, (3, 3), activation='sigmoid', kernel_initializer='he_uniform', padding='same')(x)
        model = models.Model(inputs=input_img, outputs=y)

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=[metrics.Accuracy()]
        )
    return model



### RNN ###
def loadSampleRnn(input_shape=(5,256,256,3),weights_path=None):
    with strategy.scope():
        inputs = layers.Input(shape=input_shape)
        x0 = layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", return_sequences=True, data_format="channels_last")(inputs)
        x0 = layers.BatchNormalization(momentum=0.6)(x0)
        x0 = layers.ConvLSTM2D(filters=16, kernel_size=(3,3), padding="same", return_sequences=True, data_format="channels_last")(x0)
        x0 = layers.BatchNormalization(momentum=0.8)(x0)
        x0 = layers.ConvLSTM2D(filters=3, kernel_size=(3,3), padding="same", return_sequences=False, data_format="channels_last")(x0)
        x0 = layers.Flatten()(x0)
        # x0 = layers.Dense(4096, activation='relu')(x0)
        # x0 = layers.Dense(2048, activation='relu')(x0)
        x0 = layers.Dense(512, activation='relu')(x0)
        x0 = layers.Dense(128, activation='relu')(x0)
        output = layers.Dense(1, activation='sigmoid')(x0)
        # output = layers.Activation('tanh')(x0)
        model = models.Model(inputs=inputs, outputs=output)

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=[metrics.Accuracy(), metrics.AUC(), metrics.Precision(), metrics.Recall() , metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives()]
        )
    return model





### jsonパラメータ保存 ###
def saveParams(params,filename="./params.json"):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=False, ensure_ascii=False)


### jsonパラメータ読込 ###
def loadParams(filename="./params.json"):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


### モデルテスト ###
def testModel(model,test_generator):
    print(f"START TEST: {datetime.datetime.now()}")
    loss_and_metrics = model.evaluate_generator(
        test_generator
    )
    print("Test loss:",loss_and_metrics[0])
    print("Test accuracy:",loss_and_metrics[1])
    print("Test AUC:",loss_and_metrics[2])
    print("Test Precision:",loss_and_metrics[3])
    print("Test Recall:",loss_and_metrics[4])
    print("Test TP:",loss_and_metrics[5])
    print("Test TN:",loss_and_metrics[6])
    print("Test FP:",loss_and_metrics[7])
    print("Test FN:",loss_and_metrics[8])
    print(f"FINISH TEST: {datetime.datetime.now()}")


### モデル保存 ###
def saveModel(model,model_dir):
    print(f"START MODEL SAVE: {datetime.datetime.now()}")
    try:
        model.save(f'{model_dir}/model.h5')
    except NotImplementedError:
        print('Error')
    model.save_weights(f'{model_dir}/weight.hdf5')
    print(f"FINISH MODEL SAVE: {datetime.datetime.now()}")


### グラフ作成 ###
def makeGraph(history,model_dir):
    try:
        fig = plt.figure()
        plt.plot(range(1, len(history['accuracy'])+1), history['accuracy'], "-o")
        plt.plot(range(1, len(history['val_accuracy'])+1), history['val_accuracy'], "-o")
        plt.title('Model accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.legend(['accuracy','val_accuracy'], loc='best')
        fig.savefig(model_dir+"/result.png")
        # plt.show()
    except NameError:
        print("The graph could not be saved because the process was interrupted.")


### エポック毎にhistory保存 ###
class saveHistory(callbacks.Callback):
    def __init__(self,filepath="history.json"):
        self.filepath = filepath
        self.history = {
            "loss":[],
            "accuracy":[],
            "auc":[],
            "precision":[],
            "recall":[],
            "true_positives":[],
            "true_negatives":[],
            "false_positives":[],
            "false_negatives":[],
            "val_loss":[],
            "val_accuracy":[],
            "val_auc":[],
            "val_precision":[],
            "val_recall":[],
            "val_true_positives":[],
            "val_true_negatives":[],
            "val_false_positives":[],
            "val_false_negatives":[],
        }
    
    def on_train_begin(self, logs={}):
        if os.path.isfile(self.filepath):
            with open(self.filepath, 'r') as f:
                self.history = json.load(f)
        # else:
        #     with open(self.filepath, 'w') as f:
        #         json.dump(self.history, f, indent=2, sort_keys=False, ensure_ascii=False)

    def on_epoch_end(self, epoch, logs={}):
        self.history["loss"].append(logs.get('loss'))
        self.history["accuracy"].append(logs.get('accuracy'))
        self.history["auc"].append(logs.get('auc'))
        self.history["precision"].append(logs.get('precision'))
        self.history["recall"].append(logs.get('recall'))
        self.history["true_positives"].append(logs.get('true_positives'))
        self.history["true_negatives"].append(logs.get('true_negatives'))
        self.history["false_positives"].append(logs.get('false_positives'))
        self.history["false_negatives"].append(logs.get('false_negatives'))
        self.history["val_loss"].append(logs.get('val_loss'))
        self.history["val_accuracy"].append(logs.get('val_accuracy'))
        self.history["val_auc"].append(logs.get('val_auc'))
        self.history["val_precision"].append(logs.get('val_precision'))
        self.history["val_recall"].append(logs.get('val_recall'))
        self.history["val_true_positives"].append(logs.get('val_true_positives'))
        self.history["val_true_negatives"].append(logs.get('val_true_negatives'))
        self.history["val_false_positives"].append(logs.get('val_false_positives'))
        self.history["val_false_negatives"].append(logs.get('val_false_negatives'))
        with open(self.filepath, 'w') as f:
            json.dump(self.history, f, indent=2, sort_keys=False, ensure_ascii=False)
