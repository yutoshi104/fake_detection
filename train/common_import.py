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

import tensorflow as tf
tf_version = tf.__version__
print("tf version: "+tf_version)
# 2.3.1
# 2.7.0
# 2.9.1

import tensorflow.python.keras as keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
# from tensorflow.python.keras import applications
from tensorflow.keras import applications
from tensorflow.keras import optimizers
# from keras import optimizers
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import metrics
if tf_version=="2.3.1":
    from tensorflow.python.keras.datasets import cifar10
    from tensorflow.python.keras.datasets import mnist
else:
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.datasets import mnist
if tf_version=="2.3.1":
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
else:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
import sys
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
try:
    from ImageIterator import *
except ImportError as e:
    print(e)
    from ImageIterator_nocv2 import *
try:
    from ImageSequenceIterator import *
except ImportError as e:
    print(e)
    from ImageSequenceIterator_nocv2 import *

### 環境変数読み込み ###
from dotenv import load_dotenv
load_dotenv()

### GPU稼働確認 ###
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

strategy = tf.distribute.MirroredStrategy(tf.config.list_logical_devices('GPU'))
    

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
    
    print(type(X_train))
    print(X_train.shape)
    print(type(X_test))
    print(X_test.shape)
    print(type(y_train))
    print(y_train.shape)
    print(y_train[0])
    print(y_train[1])
    print(np.count_nonzero(y_train==0))
    print(np.count_nonzero(y_train==1))
    print(type(y_test))
    print(y_test.shape)
    print(y_test[0])
    print(y_test[1])
    print(np.count_nonzero(y_test==0))
    print(np.count_nonzero(y_test==1))
    exit()

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






####################################################################################################
# CNNモデル構造
####################################################################################################


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

### CNN Xception Dropoutあり ###
def loadXceptionDropout(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="XceptionDrop")
        model.add(applications.xception.Xception(include_top=False, weights=None, input_shape=input_shape))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(512, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
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


### CNN EfficientNetV2B0 Dropoutあり ###
def loadEfficientNetV2B0bbb(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="EfficientNetV2B0")
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-b0', weights=None, include_top=False))
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
            optimizer="sgd",
            metrics=getMetrics("all")
        )
    return model

### CNN EfficientNetV2B3 Dropoutあり ###
def loadEfficientNetV2B3(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="EfficientNetV2B3")
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-b3', weights=None, include_top=False))
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer="sgd",
            metrics=getMetrics("all")
        )
    return model

### CNN EfficientNetV2S Dropoutあり ###
def loadEfficientNetV2S(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="EfficientNetV2S")
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-s', weights=None, include_top=False))
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer="sgd",
            metrics=getMetrics("all")
        )
    return model

### CNN EfficientNetV2L Dropoutあり ###
def loadEfficientNetV2L(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="EfficientNetV2L")
        model.add(layers.InputLayer(input_shape=input_shape))
        model.add(efficientnetv2.effnetv2_model.get_model('efficientnetv2-l', weights=None, include_top=False))
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer="sgd",
            metrics=getMetrics("all")
        )
    return model





### CNN NASNet(large) ###
def loadNasNetLarge(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="NASNetLarge")
        model.add(applications.nasnet.NASNetLarge(include_top=False, weights=None, input_shape=input_shape))
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

### CNN NASNet(mobile) ###
def loadNasNetMobile(input_shape=(480,640,3),weights_path=None):
    with strategy.scope():
        model = models.Sequential(name="NASNetMobile")
        model.add(applications.nasnet.NASNetMobile(include_top=False, weights=None, input_shape=input_shape))
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

####################################################################################################









####################################################################################################
# LightInceptionNet
####################################################################################################


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


def original_net2(inputs,output_filter=128):
    cell_num = 8
    x = inputs
    cell1 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell2 = layers.MaxPool2D(pool_size=(2,2),strides=1,padding="same")(x)
    cell2 = layers.Conv2D(output_filter//cell_num,1,padding="same")(cell2)
    cell3 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell3 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell3)
    cell4 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell4 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell4)
    cell4 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell4)
    cell5 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell5 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell5)
    cell5 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell5)
    cell5 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell5)
    cell6 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell6 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell6)
    cell6 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell6)
    cell6 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell6)
    cell6 = layers.Conv2D(output_filter//cell_num,9,padding="same")(cell6)
    cell7 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell7 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell7)
    cell7 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell7)
    cell7 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell7)
    cell7 = layers.Conv2D(output_filter//cell_num,9,padding="same")(cell7)
    cell7 = layers.Conv2D(output_filter//cell_num,11,padding="same")(cell7)
    cell8 = layers.Conv2D(output_filter//cell_num,1,padding="same")(x)
    cell8 = layers.Conv2D(output_filter//cell_num,3,padding="same")(cell8)
    cell8 = layers.Conv2D(output_filter//cell_num,5,padding="same")(cell8)
    cell8 = layers.Conv2D(output_filter//cell_num,7,padding="same")(cell8)
    cell8 = layers.Conv2D(output_filter//cell_num,9,padding="same")(cell8)
    cell8 = layers.Conv2D(output_filter//cell_num,11,padding="same")(cell8)
    cell8 = layers.Conv2D(output_filter//cell_num,13,padding="same")(cell8)
    x = layers.Concatenate()([cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8])

    x = layers.SeparableConv2D(output_filter, (3,3), strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


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


####################################################################################################






####################################################################################################
# LightInceptionNetの改良（By MySelf）
####################################################################################################


### pattern 1 ###
# def loadOriginalUpdate1(input_shape=(480,640,3),weights_path=None):
#     with strategy.scope():
#         def original_net(inputs,output_filter=128):
#             cell_num = 8
#             x = inputs

#             red = layers.SeparableConv2D(output_filter//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(x)
#             cell1 = red
#             cell2 = layers.AveragePooling2D(pool_size=(2,2),strides=1,padding="same")(x)
#             cell2 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell2)
#             cell3 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
#             cell4 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
#             cell5 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
#             cell5 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell5)
#             cell6 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
#             cell6 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell6)
#             cell7 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
#             cell7 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell7)
#             cell7 = layers.SeparableConv2D(output_filter//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell7)
#             cell8_residual = layers.SeparableConv2D(output_filter//2//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(red)
#             cell8 = layers.SeparableConv2D(output_filter//2//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(red)
#             cell8 = layers.Concatenate()([cell8_residual,cell8])
#             cell8_residual = layers.SeparableConv2D(output_filter//2//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(cell8)
#             cell8 = layers.SeparableConv2D(output_filter//2//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell8)
#             cell8 = layers.Concatenate()([cell8_residual,cell8])
#             cell8_residual = layers.SeparableConv2D(output_filter//2//cell_num,(1,1),padding="same", kernel_initializer='he_uniform')(cell8)
#             cell8 = layers.SeparableConv2D(output_filter//2//cell_num,(3,3),padding="same", kernel_initializer='he_uniform')(cell8)
#             cell8 = layers.Concatenate()([cell8_residual,cell8])
#             x = layers.Concatenate()([cell1,cell2,cell3,cell4,cell5,cell6,cell7,cell8])
#             return x

#         inputs = keras.Input(shape=input_shape)
#         x = inputs

#         ###1つ目###
#         x = layers.Conv2D(32, (3,3), strides=2, padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2D(64, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         # (128,128,64)

#         residual = x
#         residual = layers.Conv2D(128, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
#         residual = layers.BatchNormalization()(residual)
#         x = layers.SeparableConv2D(128, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(128, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
#         x = layers.add([x, residual])
#         x = layers.Activation('relu')(x)
#         residual = layers.Conv2D(256, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
#         residual = layers.BatchNormalization()(residual)
#         x = layers.SeparableConv2D(256, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(256, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
#         x = layers.add([x, residual])
#         x = layers.Activation('relu')(x)
#         residual = layers.Conv2D(512, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
#         residual = layers.BatchNormalization()(residual)
#         x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
#         x = layers.add([x, residual])
#         x = layers.Activation('relu')(x)
#         # (16,16,512)

#         x = layers.SeparableConv2D(512, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         for i in range(5):
#             residual = x
#             x = layers.BatchNormalization()(x)
#             x = layers.Activation('relu')(x)
#             x = original_net(x,512)
#             x = layers.BatchNormalization()(x)
#             x = layers.Activation('relu')(x)
#             x = original_net(x,512)
#             x = layers.BatchNormalization()(x)
#             x = layers.Activation('relu')(x)
#             x = layers.Dropout(rate=0.3)(x)
#             x = original_net(x,512)
#             x = layers.add([x, residual])
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         # (16,16,512)


#         residual = x
#         residual = layers.Conv2D(768, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
#         residual = layers.BatchNormalization()(residual)
#         x = layers.SeparableConv2D(768, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(768, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
#         x = layers.add([x, residual])
#         x = layers.Activation('relu')(x)
#         residual = x
#         x = original_net(x,768)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = original_net(x,768)
#         x = layers.BatchNormalization()(x)
#         x = layers.add([x, residual])
#         x = layers.Activation('relu')(x)
#         # (8,8,768)

#         residual = x
#         residual = layers.Conv2D(1024, (1,1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
#         residual = layers.BatchNormalization()(residual)
#         x = layers.SeparableConv2D(1024, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(1024, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
#         x = layers.add([x, residual])
#         # (4,4,1024)

#         x = layers.SeparableConv2D(1536, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(2048, (3,3), padding="same", kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(rate=0.5)(x)
#         # (4,4,2048)


#         x = layers.GlobalAveragePooling2D()(x)
#         # x = layers.Flatten()(x)
#         x = layers.Dense(256, kernel_initializer='he_uniform')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Dropout(0.5)(x)
#         x = layers.Dense(1, activation='sigmoid')(x)

#         model = models.Model(inputs=inputs, outputs=x, name="OriginalNet")

#         if weights_path != None:
#             model.load_weights(weights_path)
#         model.compile(
#             loss='binary_crossentropy',
#             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#             metrics=getMetrics("all")
#         )
#     return model


####################################################################################################








####################################################################################################
# LightInceptionNetの改良（By ChatGPT）
####################################################################################################

### pattern 1 ###

def loadOriginalGptUpdate1(input_shape=(256, 256, 3), weights_path=None):
    with strategy.scope():
        inputs = keras.Input(shape=input_shape)
        x = inputs

        # モジュールの複数回使用
        x = layers.Conv2D(32, (3, 3), strides=2, padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        residual = x
        residual = layers.Conv2D(128, (1, 1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

        residual = layers.Conv2D(256, (1, 1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

        residual = layers.Conv2D(512, (1, 1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(512, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(512, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

        x = layers.SeparableConv2D(512, (3, 3), padding="same", kernel_initializer='he_uniform')(x)

        for _ in range(5):
            residual = x
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = original_module(x, 512)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = original_module(x, 512)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(rate=0.3)(x)
            x = original_module(x, 512)
            x = layers.add([x, residual])

        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        residual = layers.Conv2D(768, (1, 1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(768, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(768, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

        residual = x
        x = original_module(x, 768)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = original_module(x, 768)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)

        residual = layers.Conv2D(1024, (1, 1), strides=2, padding="same", kernel_initializer='he_uniform')(residual)
        residual = layers.BatchNormalization()(residual)
        x = layers.SeparableConv2D(1024, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(1024, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv2D(1536, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(2048, (3, 3), padding="same", kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(rate=0.5)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x, name="ImprovedNet")

        if weights_path is not None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=getMetrics("all")
        )

    return model




####################################################################################################






####################################################################################################
# AutoEncoder
####################################################################################################


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


####################################################################################################








####################################################################################################
# RNNモデル構造
####################################################################################################


### SampleRNN ###
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
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=getMetrics("all")
        )
    return model


### ConvLSTM ###
def loadConvLSTM(input_shape=(5,256,256,3),weights_path=None):
    with strategy.scope():
        inputs = layers.Input(shape=input_shape)
        x = layers.ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True, data_format="channels_last")(inputs)
        x = layers.BatchNormalization(momentum=0.6)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=True, data_format="channels_last")(x)
        x = layers.BatchNormalization(momentum=0.7)(x)
        x = layers.Dropout(0.5)(x)
        # x = layers.ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=True, data_format="channels_last")(x)
        # x = layers.BatchNormalization(momentum=0.8)(x)
        # x = layers.Dropout(0.5)(x)
        # x = layers.ConvLSTM2D(128, (3, 3), activation='relu', padding='same', return_sequences=True, data_format="channels_last")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=output, name="ConvLSTM")

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(lr=1e-4),
            metrics=getMetrics("all")
        )
    return model


### SampleViT(使えない) ###
def loadSampleViT(input_shape=(5,256,256,3),weights_path=None):
    with strategy.scope():

        num_frames, height, width, channels = input_shape
        # Transformer parameters
        embedding_dim = 64
        num_heads = 4
        ff_dim = 128
        # Encoder
        inputs = layers.Input(shape=(num_frames, height, width, channels))
        x = layers.Reshape((num_frames, height * width, channels))(inputs)
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)(x, x)
        x = layers.Dropout(0.5)(x)
        print(x.shape)
        x = layers.Reshape((num_frames, height, width, channels))(x)
        print(x.shape)
        # Convolutional layers
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        print(x.shape)
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        print(x.shape)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(0.5)(x)
        # Classifier
        x = layers.Flatten()(x)
        x = layers.Dense(ff_dim, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        # Model
        model = models.Model(inputs, outputs, name="ViT")

        if weights_path != None:
            model.load_weights(weights_path)
        model.compile(
            loss='binary_crossentropy',
            # optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            optimizer=optimizers.Adam(learning_rate=1e-4),
            metrics=getMetrics("all")
        )
    return model


####################################################################################################









####################################################################################################
# 学習・テスト用
####################################################################################################


### jsonパラメータ保存 ###
def saveParams(params,filename="./params.json"):
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=False, ensure_ascii=False)


### jsonパラメータ読込 ###
def loadParams(filename="./params.json"):
    with open(filename, 'r') as f:
        params = json.load(f)
    return params


### モデルテスト(画像単位) ###
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
    return loss_and_metrics


### AutoKerasのモデルテスト(画像単位) ###
def testModelForAk(clf,test_dataset):
    print(f"START TEST: {datetime.datetime.now()}")
    loss_and_metrics = clf.evaluate(
        test_dataset
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
    return loss_and_metrics


### モデルテスト(動画単位) (未実装) ###
def testModel_Movie(model,test_generator):
    print(f"START TEST (Movie): {datetime.datetime.now()}")

    exit()
    # idごとにgeneratorを作成→リストに格納
    # for generator_list:
        # evaluate(generator)
        # フェイクが1枚でもあったらフェイク判定

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
    print(f"FINISH TEST (Movie): {datetime.datetime.now()}")


### モデル保存 ###
def saveModel(model,model_dir):
    print(f"START MODEL SAVE: {datetime.datetime.now()}")
    try:
        model.save(f'{model_dir}/model.h5')
    except NotImplementedError:
        print('Error')
    model.save_weights(f'{model_dir}/weight.hdf5')
    print(f"FINISH MODEL SAVE: {datetime.datetime.now()}")


### AutoKerasのモデル保存 ###
def saveModelForAk(clf,model_dir):
    print(f"START MODEL SAVE: {datetime.datetime.now()}")
    model = clf.export_model()
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


### グラフ作成(ROC曲線) ###
def makeROC(history,model_dir):
    try:
        pass
    except NameError:
        print("The graph could not be saved because the process was interrupted.")


### エポック毎にhistory保存 ###
class saveHistory(callbacks.Callback):
    def __init__(self,filepath="history.json"):
        self.epoch_start_time = None
        self.filepath = filepath
        self.history = {
            "elapsed_time":[],
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

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        if "elapsed_time" not in self.history: # 追記したものなので、しばらくはこれで
            self.history["elapsed_time"] = []
        self.history["elapsed_time"].append(time.time()-self.epoch_start_time)
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


####################################################################################################










####################################################################################################
# CNN用データ生成 (image)
####################################################################################################


### Celebのimagepathリスト取得 ###
def makeImagePathList_Celeb(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data',
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        validation_rate=0.1,
        test_rate=0.1,
        data_type=None
    ):
    class_file_num = {}
    class_weights = {}
    train_data = []
    validation_data = []
    test_data = []
    train_rate = 1 - validation_rate - test_rate
    s1 = (int)(59*train_rate)
    s2 = (int)(59*(train_rate+validation_rate))
    id_list = list(range(62))
    id_list.remove(14)
    id_list.remove(15)
    id_list.remove(18)
    # random.shuffle(id_list)
    train_id_list = id_list[ : s1]
    validation_id_list = id_list[s1 : s2]
    test_id_list = id_list[s2 : ]
    print("\tTRAIN IMAGE DATA ID: ",end="")
    print(train_id_list)
    print("\tVALIDATION IMAGE DATA ID: ",end="")
    print(validation_id_list)
    print("\tTEST IMAGE DATA ID: ",end="")
    print(test_id_list)
    del id_list
    data_num = 0
    for l,c in enumerate(classes):
        image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
        path_num = len(image_path_list)
        data_num += path_num
        regexp = r'^.+?id(?P<id>(\d+))(_id(?P<id2>\d+))?_(?P<key>\d+)_(?P<num>\d+).(?P<ext>.{2,4})$'
        past_path = image_path_list[0]
        movie_image_list = []
        for i in range(1,len(image_path_list)):
            past_ids = re.search(regexp,past_path).groupdict()
            now_ids = re.search(regexp,image_path_list[i]).groupdict()
            if (past_ids['id']==now_ids['id']) and (past_ids['id2']==None or past_ids['id2']==now_ids['id2']) and (past_ids['key']==now_ids['key']):
                movie_image_list.append([image_path_list[i],l])
            else:
                if int(past_ids['id']) in train_id_list:
                    train_data.append(movie_image_list)
                elif int(past_ids['id']) in validation_id_list:
                    validation_data.append(movie_image_list)
                elif int(past_ids['id']) in test_id_list:
                    test_data.append(movie_image_list)
                movie_image_list = []
                movie_image_list.append([image_path_list[i],l])
            past_path = image_path_list[i]
        # 不均衡データ調整
        class_file_num[c] = path_num
        if l==0:
            n = class_file_num[c]
        class_weights[l] = 1 / (class_file_num[c]/n)

    train_data = list(chain.from_iterable(train_data))
    validation_data = list(chain.from_iterable(validation_data))
    test_data = list(chain.from_iterable(test_data))
    if data_type=="train":
        return train_data
    elif data_type=="validation":
        return validation_data
    elif data_type=="test":
        return test_data
    else:
        return (train_data, validation_data, test_data, data_num, class_file_num, class_weights)


### Celebのimage_generator作成 ###
def makeImageDataGenerator_Celeb(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data',
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        validation_rate=0.1,
        test_rate=0.1,
        batch_size=32,
        image_size=(256,256,3),
        rotation_range=15.0,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.1,
        channel_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        edge=None
    ):
    if image_size[2]==1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'

    train_data, validation_data, test_data, data_num, class_file_num, class_weights = makeImagePathList_Celeb(
        data_dir=data_dir,
        classes=classes,
        validation_rate=validation_rate,
        test_rate=test_rate,
    )

    train_data_num = len(train_data)
    validation_data_num = len(validation_data)
    test_data_num = len(test_data)
    print("\tALL IMAGE PATH NUM: " + str(data_num))
    print("\tALL DATA NUM: " + str(train_data_num+validation_data_num+test_data_num))
    print("\tTRAIN DATA NUM: " + str(train_data_num))
    print("\tVALIDATION DATA NUM: " + str(validation_data_num))
    print("\tTEST DATA NUM: " + str(test_data_num))
    def makeGenerator(data,subset="training"):
        return ImageIterator(
            data,
            batch_size=batch_size,
            target_size=image_size[:2],
            color_mode=color_mode,
            seed=1,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            edge=edge,
            rescale=1./255,
            data_format='channels_last',
            subset=subset)
    train_generator = makeGenerator(train_data,"training")
    validation_generator = makeGenerator(validation_data,"validation")
    test_generator = makeGenerator(test_data,"test")
    del train_data
    del validation_data
    del test_data
    return (train_generator,validation_generator,test_generator,class_file_num,class_weights)


### CelebのImageGenerator作成(AutoKeras用) ###
def makeImageDataGenerator_Celeb_ForAk(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data',
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        validation_rate=0.1,
        test_rate=0.1,
        batch_size=32,
        image_size=(256,256,3),
        rotation_range=15.0,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.1,
        channel_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        edge=None
    ):

    if image_size[2]==1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'

    train_data, validation_data, test_data, data_num, class_file_num, class_weights = makeImagePathList_Celeb(
        data_dir=data_dir,
        classes=classes,
        validation_rate=validation_rate,
        test_rate=test_rate,
    )

    train_images = tf.data.Dataset.from_tensor_slices(np.array(train_data)[:,0])
    validation_images = tf.data.Dataset.from_tensor_slices(np.array(validation_data)[:,0])
    test_images = tf.data.Dataset.from_tensor_slices(np.array(test_data)[:,0])
    train_labels = tf.data.Dataset.from_tensor_slices(np.array(train_data)[:,1])
    validation_labels = tf.data.Dataset.from_tensor_slices(np.array(validation_data)[:,1])
    test_labels = tf.data.Dataset.from_tensor_slices(np.array(test_data)[:,1])
    
    train_images = train_images.map(
        lambda img_path: preprocess(img_path, image_size)
    )
    validation_images = validation_images.map(
        lambda img_path: preprocess(img_path, image_size)
    )
    test_images = test_images.map(
        lambda img_path: preprocess(img_path, image_size)
    )
    train_dataset = tf.data.Dataset.zip((train_images, train_labels))
    validation_dataset = tf.data.Dataset.zip((validation_images, validation_labels))
    test_dataset = tf.data.Dataset.zip((test_images, test_labels))
    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = validation_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return (train_dataset,validation_dataset,test_dataset,class_file_num,class_weights)


####################################################################################################





####################################################################################################
# RNN用データ生成 (sequence)
####################################################################################################


### Celebのsequencepathリスト取得 ###
def makeSequencePathList_Celeb(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data',
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        validation_rate=0.1,
        test_rate=0.1,
        nt=10,
        per_frame=2,
        data_type=None
    ):
    class_file_num = {}
    class_weights = {}
    train_data = []
    validation_data = []
    test_data = []
    train_rate = 1 - validation_rate - test_rate
    s1 = (int)(59*train_rate)
    s2 = (int)(59*(train_rate+validation_rate))
    id_list = list(range(62))
    id_list.remove(14)
    id_list.remove(15)
    id_list.remove(18)
    # random.shuffle(id_list)
    train_id_list = id_list[ : s1]
    validation_id_list = id_list[s1 : s2]
    test_id_list = id_list[s2 : ]
    print("\tTRAIN IMAGE DATA ID: ",end="")
    print(train_id_list)
    print("\tVALIDATION IMAGE DATA ID: ",end="")
    print(validation_id_list)
    print("\tTEST IMAGE DATA ID: ",end="")
    print(test_id_list)
    del id_list
    data_num = 0
    for l,c in enumerate(classes):
        image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
        path_num = len(image_path_list)
        regexp = r'^.+id(?P<id>(\d+))_(?P<id2>\d+)_?(?P<num>\d+).(?P<ext>.{2,4})$'
        past_path = image_path_list[0]
        i = 0
        num = 0
        while path_num > i+nt:
            sequence_path_list = []
            if (int(re.search(regexp,image_path_list[i]).groupdict()['num'])%per_frame)==0:
                for j in range(nt):
                    past_ids = re.search(regexp,past_path).groupdict()
                    now_ids = re.search(regexp,image_path_list[i+j]).groupdict()
                    if (past_ids['id']==now_ids['id']) and (past_ids['id2']==now_ids['id2']):
                        sequence_path_list.append(image_path_list[i+j])
                    else:
                        i += j
                        past_path = image_path_list[i]
                        break
                    past_path = image_path_list[i+j]
                else:
                    if int(past_ids['id']) in train_id_list:
                        train_data.append([[sequence_path_list,l]])
                    elif int(past_ids['id']) in validation_id_list:
                        validation_data.append([[sequence_path_list,l]])
                    elif int(past_ids['id']) in test_id_list:
                        test_data.append([[sequence_path_list,l]])
                    num += 1
                    i += 1
            else:
                i += 1
        data_num += num
        # 不均衡データ調整
        class_file_num[c] = num
        if l==0:
            n = class_file_num[c]
        class_weights[l] = 1 / (class_file_num[c]/n)
    train_data = list(chain.from_iterable(train_data))
    validation_data = list(chain.from_iterable(validation_data))
    test_data = list(chain.from_iterable(test_data))
    if data_type=="train":
        return train_data
    elif data_type=="validation":
        return validation_data
    elif data_type=="test":
        return test_data
    else:
        return (train_data, validation_data, test_data, data_num, class_file_num, class_weights)


### Celebのsequence_generator作成 ###
def makeSequenceDataGenerator_Celeb(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data',
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        validation_rate=0.1,
        test_rate=0.1,
        batch_size=32,
        image_size=(256,256,3),
        nt=50,
        per_frame=50,
        rotation_range=15.0,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.1,
        channel_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        edge=None
    ):
    if image_size[2]==1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'

    train_data, validation_data, test_data, data_num, class_file_num, class_weights = makeSequencePathList_Celeb(
        data_dir=data_dir,
        classes=classes,
        validation_rate=validation_rate,
        test_rate=test_rate,
    )

    train_data_num = len(train_data)
    validation_data_num = len(validation_data)
    test_data_num = len(test_data)
    print("\tALL IMAGE PATH NUM: " + str(data_num))
    print("\tALL DATA NUM: " + str(train_data_num+validation_data_num+test_data_num))
    print("\tTRAIN DATA NUM: " + str(train_data_num))
    print("\tVALIDATION DATA NUM: " + str(validation_data_num))
    print("\tTEST DATA NUM: " + str(test_data_num))
    def makeGenerator(data,subset="training"):
        return ImageSequenceIterator(
            data,
            batch_size=batch_size,
            target_size=image_size[:2],
            nt=nt,
            color_mode=color_mode,
            seed=1,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            edge=edge,
            rescale=1./255,
            data_format='channels_last',
            subset=subset)
    train_generator = makeGenerator(train_data,"training")
    validation_generator = makeGenerator(validation_data,"validation")
    test_generator = makeGenerator(test_data,"test")
    del train_data
    del validation_data
    del test_data
    return (train_generator,validation_generator,test_generator,class_file_num,class_weights)


####################################################################################################









####################################################################################################
# データ生成 (汎用)
####################################################################################################


### 任意のディレクトリのgenerator作成 ###
def makeImageDataGenerator(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data',
        classes=['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90'],
        batch_size=32,
        image_size=(256,256,3),
        rotation_range=15.0,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.1,
        channel_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        edge=None
    ):
    if image_size[2]==1:
        color_mode = 'grayscale'
    else:
        color_mode = 'rgb'
    class_file_num = {}
    class_weights = {}
    data = []
    data_num = 0
    for l,c in enumerate(classes):
        image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
        path_num = len(image_path_list)
        data_num += path_num
        for i,image_path in enumerate(image_path_list):
            data.append([[image_path,l]])
        # 不均衡データ調整
        class_file_num[c] = path_num
        if l==0:
            n = class_file_num[c]
        class_weights[l] = 1 / (class_file_num[c]/n)

    data = list(chain.from_iterable(data))
    print("\tALL DATA NUM: " + str(data_num))
    def makeGenerator(data,subset="test"):
        return ImageIterator(
            data,
            batch_size=batch_size,
            target_size=image_size[:2],
            color_mode=color_mode,
            seed=1,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            edge=edge,
            rescale=1./255,
            data_format='channels_last',
            subset=subset)
    generator = makeGenerator(data,"test")
    del data
    return (generator,class_file_num,class_weights)


### Celebのパス取得(train,validation,test) ###
def getPathList_Celeb(
        data_dir='/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-real-image-face-90',
        validation_rate=0.1,
        test_rate=0.1,
        data_type="test"
    ):
    train_data = []
    validation_data = []
    test_data = []
    train_rate = 1 - validation_rate - test_rate
    s1 = (int)(59*train_rate)
    s2 = (int)(59*(train_rate+validation_rate))
    id_list = list(range(62))
    id_list.remove(14)#14
    id_list.remove(15)#15
    id_list.remove(18)
    # random.shuffle(id_list)
    train_id_list = id_list[ : s1]
    validation_id_list = id_list[s1 : s2]
    test_id_list = id_list[s2 : ]
    del id_list
    image_path_list = sorted(glob.glob(data_dir+"/*"))
    regexp = r'^.+?id(?P<id>(\d+))(_id(?P<id2>\d+))?_(?P<key>\d+)_(?P<num>\d+).(?P<ext>.{2,4})$'
    for i,image_path in enumerate(image_path_list):
        ids = re.search(regexp,image_path).groupdict()
        if int(ids['id']) in train_id_list:
            train_data.append([str(image_path)])
        elif int(ids['id']) in validation_id_list:
            validation_data.append([str(image_path)])
        elif int(ids['id']) in test_id_list:
            test_data.append([str(image_path)])

    train_data = list(chain.from_iterable(train_data))
    validation_data = list(chain.from_iterable(validation_data))
    test_data = list(chain.from_iterable(test_data))
    if data_type=="train":
        return train_data
    elif data_type=="validation":
        return validation_data
    elif data_type=="test":
        return test_data
    else:
        return (train_data,validation_data,test_data)
    
    
####################################################################################################
    











####################################################################################################
# 画像前処理
####################################################################################################


def preprocess(file_path,image_size=(256,256,3)):
    raw = tf.io.read_file(file_path)
    image = tf.io.decode_image(
        raw, channels=image_size[2], expand_animations=False
    )
    image = tf.image.resize(image, image_size[:2], method='nearest')
    image.set_shape((image_size[0], image_size[1], image_size[2]))
    ###今後前処理を実装###
    return image
