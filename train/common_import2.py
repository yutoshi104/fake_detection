import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import metrics
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import tensorflow.keras.backend as K
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
#import cv2
import numpy as np
import signal
from pathlib import Path
import glob
import cv2
import pickle
import re
import time
import datetime
import os
import random
import copy
from random import shuffle
from itertools import islice,chain
from pprint import pprint

from tensorflow.python.util.nest import _yield_value

from defined_models import efficientnetv2
from originalnet import *
from ImageIterator import *
from ImageSequenceIterator import *

from common_import import *


def InceptionV3_block1(x):
    # def f(x):
    b1 = layers.Convolution2D(64, (1,1), strides=2, padding='same')(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)

    b2 = layers.Convolution2D(48, (1,1))(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    b2 = layers.Convolution2D(96, (3,3), strides=2, padding='same')(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)


    b3 = layers.AveragePooling2D(pool_size=(3, 3), strides=2,  padding='same')(x)
    b3 = layers.Convolution2D(64, (3,3), padding='same')(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)

    b4 = layers.Convolution2D(64, (1,1))(x)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)
    b4 = layers.Convolution2D(96, (3,3), padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)
    b4 = layers.Convolution2D(32, (3,3),strides=2, padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)

    output = layers.concatenate([b1, b2, b3, b4], axis=-1)
    return output
    # return f


def InceptionV3_block2(x):
    # def f(x):
    b1 = layers.Convolution2D(64, (1,1), strides=2,  padding='same')(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)

    b2 = layers.Convolution2D(48, (1,1))(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)
    b2 = layers.Convolution2D(96, (3,3), strides=2,  padding='same')(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)


    b3 = layers.AveragePooling2D(pool_size=(3, 3), strides=2,  padding='same')(x)
    b3 = layers.Convolution2D(64, (3,3), padding='same')(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)

    b4 = layers.Convolution2D(64, (1,1))(x)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)
    b4 = layers.Convolution2D(96, (3,3), padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)
    b4 = layers.Convolution2D(64, (3,3),strides=2, padding='same')(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)

    output = layers.concatenate([b1, b2, b3, b4], axis=-1)
    return output
    # return f


### CNN Inception Original ###
def loadInceptionV3Original(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = keras.Input(shape=input_shape)

        x = layers.Convolution2D(32, (1,1), strides=2)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

        x = InceptionV3_block1(x)
        x = InceptionV3_block2(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, kernel_initializer='he_uniform')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = models.Model(inputs=inputs, outputs=x, name="InceptionV3")

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model


### CNN DenseNet121 ###
def loadDenseNet121(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="DenseNet121")
        model.add(applications.densenet.DenseNet121(include_top=False, weights=None, input_shape=input_shape))
        print(len(applications.densenet.DenseNet121(include_top=False, weights=None, input_shape=input_shape).layers))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model


### CNN DenseNet169 ###
def loadDenseNet169(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="DenseNet169")
        model.add(applications.densenet.DenseNet169(include_top=False, weights=None, input_shape=input_shape))
        print(len(applications.densenet.DenseNet169(include_top=False, weights=None, input_shape=input_shape).layers))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model


### CNN DenseNet201 ###
def loadDenseNet201(input_shape=(480,640,3),gpu_count=2):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = models.Sequential(name="DenseNet201")
        model.add(applications.densenet.DenseNet201(include_top=False, weights=None, input_shape=input_shape))
        print(len(applications.densenet.DenseNet201(include_top=False, weights=None, input_shape=input_shape).layers))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, kernel_initializer='he_uniform'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=getMetrics("all")
        )
    return model

