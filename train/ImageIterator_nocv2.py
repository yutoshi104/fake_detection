import numpy as np
import os
import tensorflow as tf
tf_version = tf.__version__
if tf_version=="2.3.1":
    from tensorflow.python.keras.preprocessing.image import Iterator
    from tensorflow.python.keras.preprocessing.image import load_img,img_to_array
    from tensorflow.python.keras.preprocessing.image import random_rotation,random_shift,random_shear,random_zoom,apply_channel_shift,random_channel_shift,apply_brightness_shift,random_brightness,apply_affine_transform
else:
    from tensorflow.keras.preprocessing.image import Iterator
    from tensorflow.keras.preprocessing.image import load_img,img_to_array
    from tensorflow.keras.preprocessing.image import random_rotation,random_shift,random_shear,random_zoom,apply_channel_shift,random_channel_shift,apply_brightness_shift,random_brightness,apply_affine_transform
from tensorflow.python.keras.backend import dtype


"""
引数：
	画像データのpathのリスト
		[
			["/aaa/bbb/ccc/ddd.png",0],
			["/aaa/bbb/ccc/eee.png",1],
			["/aaa/bbb/ccc/fff.png",0],
			["/aaa/bbb/ccc/ggg.png",2],
		]
"""
class ImageIterator(Iterator):

    def __init__(
            self,
            data,
            batch_size=32,
            target_size=(256,256),
            color_mode='rgb',
            shuffle=True,
            seed=None,
            rotation_range=0.0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            brightness_range=None,
            shear_range=0.0,
            zoom_range=0.0,
            channel_shift_range=0.0,
            fill_mode='nearest',
            cval=0.0,
            horizontal_flip=False,
            vertical_flip=False,
            edge=None,
            rescale=None,
            preprocessing_function=None,
            data_format='channels_last',
            subset='train'):
        self.data = data
        self.target_size = target_size
        self.color_mode = color_mode
        self.channel = 1 if self.color_mode=="grayscale" else (3 if self.color_mode=="rgb" else 4)
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        if type(zoom_range) is not tuple:
            zoom_range = (1-zoom_range, 1+zoom_range)
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.edge_type = edge
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.data_format = data_format
        if data_format=="channels_last":
            row_axis,col_axis,channel_axis = 0,1,2
        else:
            row_axis,col_axis,channel_axis = 1,2,0
        self.row_axis = row_axis
        self.col_axis = col_axis
        self.channel_axis = channel_axis
        self.subset = subset
        n = len(data)
        super(ImageIterator, self).__init__(n,batch_size,shuffle,seed)


    def _get_batches_of_transformed_samples(self, index_array):
        if self.data_format=="channels_last":
            x = np.zeros((len(index_array),)+self.target_size+(self.channel,), dtype=np.float32)
        else:
            x = np.zeros((len(index_array),)+(self.channel,)+self.target_size, dtype=np.float32)
        y = np.zeros((len(index_array),1), dtype=np.float32)
        for i,idx in enumerate(index_array):
            path_data = self.data[idx]
            path = path_data[0]
            label = path_data[1]
            img = img_to_array(load_img(path,color_mode=self.color_mode,target_size=self.target_size),data_format=self.data_format,dtype='float32')
            if self.preprocessing_function==None:
                img = self.preprocess(img)
            else:
                img = self.preprocessing_function(img)
            # augmentationを実行
            img = self.rotate(img)
            img = self.shift(img)
            img = self.brightness(img)
            img = self.shear(img)
            img = self.zoom(img)
            img = self.channel_shift(img)
            img = self.flip(img)
            x[i] = img
            y[i][0] = label
        return x,y


    def preprocess(self,image):
        image = image * self.rescale
        return image

    def rotate(self,image):
        return random_rotation(
            image,
            self.rotation_range,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            channel_axis=self.channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval)

    def shift(self,image):
        return random_shift(
            image,
            wrg=self.width_shift_range,
            hrg=self.height_shift_range,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            channel_axis=self.channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval)

    def brightness(self,image):
        if self.brightness_range:
            image = random_brightness(
                image,
                brightness_range=self.brightness_range
            )
        return image

    def shear(self,image):
        return random_shear(
            image,
            intensity=self.shear_range,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            channel_axis=self.channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval)

    def zoom(self,image):
        return random_zoom(
            image,
            zoom_range=self.zoom_range,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            channel_axis=self.channel_axis,
            fill_mode=self.fill_mode,
            cval=self.cval)

    def channel_shift(self,image):
        return random_channel_shift(
            image,
            intensity_range=self.channel_shift_range,
            channel_axis=self.channel_axis)

    def flip(self,image):
        if self.horizontal_flip and np.random.rand()<0.5:
            image = np.fliplr(image)
        if self.vertical_flip and np.random.rand()<0.5:
            image = np.flipud(image)
        return image

    def len(self):
        return len(self.data)