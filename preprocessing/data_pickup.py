import numpy as np
import cv2
import glob
import os
import random


# dirから全てのファイルパスを取得
def getPathList(parent_dir):
    return sorted(glob.glob(parent_dir+"/*"))

# パスから画像データを取得
def getImageFromPath(img_path):
    if not os.path.exists(img_path):
        print(f'file "{img_path}" is not exist.')
        return False
    return cv2.imread(img_path)

# ぼかし処理
def image2blur(img):
    img = cv2.blur(img,(5,5))
    return img

# 画像保存(1枚)
def saveImage(data,save_path="./image.jpg"):
    cv2.imwrite(save_path, data)
    
    

import re
from pprint import pprint
from itertools import islice,chain
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
    id_list.remove(13)#14
    id_list.remove(14)#15
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


def main(data_dir,save_dir,makenum=50):
    print()
    print("data_dir: "+data_dir, end=",\t")
    print("num: "+str(makenum), end=",\t")
    print("save dir: "+save_dir)
    
    path_list = getPathList_Celeb(data_dir)
    path_list = random.sample(path_list, makenum)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_list = []
    for path in path_list:
        img = getImageFromPath(path)
        saveImage(img,save_dir+"/"+os.path.basename(path))
        img_list.append(img)

    # saveImageList(img_list,save_path=save_dir+"/concat.jpg")








if __name__=='__main__':
    random.seed(10)
    real_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-real-image-face-90"
    real_save_directory = "/home/toshi/fake_detection/data/datas/Celeb-real-image-face-90"
    fake_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-synthesis-image-face-90"
    fake_save_directory = "/home/toshi/fake_detection/data/datas/Celeb-synthesis-image-face-90"
    main(real_directory,real_save_directory,1000)
    main(fake_directory,fake_save_directory,1000)
