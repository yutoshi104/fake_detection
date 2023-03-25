# import tensorflow as tf
# from tensorflow.python.client import device_lib
# tf.get_logger().setLevel("ERROR")
# print(tf.__version__)
# print(device_lib.list_local_devices())
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

# グレースケールに変換
def convertGray(img_data):
    return cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

# エッジ抽出
def image2edgeimage(img,edge_type='sobel'):
    if edge_type=='none':
        img_edge = img
    elif edge_type=='sobel':
        img_gray = convertGray(img)
        img_edge = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    elif edge_type=='laplacian':
        img_gray = convertGray(img)
        img_edge = cv2.Laplacian(img_gray, cv2.CV_32F)
    elif edge_type=='scharr':
        img_gray = convertGray(img)
        img_edge = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=-1)
    elif edge_type=='canny':
        img_gray = convertGray(img)
        img_edge = cv2.Canny(img_gray, 100, 200).astype(np.float32)
    return img_edge

# 画像保存(1枚)
def saveImage(data,save_path="./image.jpg"):
    cv2.imwrite(save_path, data)

# 画像保存(複数枚)
def saveImageList(data_list,save_path="./concat_image.jpg"):
    l = len(data_list)
    # 埋める黒画像
    img_black = np.zeros((data_list[0].shape),dtype=np.float32)
    # 最大50枚
    if l > 50:
        data_list = data_list[:50]
        l = len(data_list)
    # 保存サイズ決定
    if l <= 10:
        h = 2
        w = 5
    elif l <= 18:
        h = 3
        w = 6
    elif l <= 28:
        h = 4
        w = 7
    elif l <= 45:
        h = 5
        w = 9
    else:
        h = 6
        w = 9
    # 高さ分の黒画像を追加
    for j in range(h):
        data_list.append(img_black)
    # 結合
    try:
        vert_list = []
        for i in range(0,l,h):
            print(data_list[0].shape)
            vert_list.append(cv2.vconcat(data_list[i:i+h]))
        concat_image = cv2.hconcat(vert_list)
        cv2.imwrite(save_path, concat_image)
    except Exception:
        print(f"Pass: {save_path}")


real_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-real-image-face-90"
fake_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-synthesis-image-face-90"
real_path_list = getPathList(real_directory)
fake_path_list = getPathList(fake_directory)
with_path_list = real_path_list + fake_path_list

def main(real_fake_with,edge_type,img_num,save_dir='concat'):
    print()
    print("data: "+real_fake_with, end=",\t")
    print("edge: "+edge_type, end=",\t")
    print("num: "+str(img_num), end=",\t")
    print("save dir: "+save_dir)
    random.seed(10)
    
    if real_fake_with=='real':
        path_list = random.sample(real_path_list, img_num)
    elif real_fake_with=='fake':
        path_list = random.sample(fake_path_list, img_num)
    elif real_fake_with=='with':
        path_list = random.sample(with_path_list, img_num)
    else:
        return False

    img_list = []
    for path in path_list:
        img = getImageFromPath(path)
        img = image2edgeimage(img,edge_type)
        print(img.dtype)
        img_list.append(img)

    if not os.path.exists(save_dir):
        os.makedirs("./"+save_dir)
    saveImageList(img_list,save_path=f"./{save_dir}/concat_{real_fake_with}_{edge_type}_{img_num}.jpg")








if __name__=='__main__':
    ###個別###
    # real_fake_with = 'with'
    # real_fake_with = 'real'
    # real_fake_with = 'fake'
    # edge_type = 'sobel'
    # img_num = 50
    # main(real_fake_with,edge_type,img_num,save_dir='individual')


    ###一括###
    real_fake_with_list = ["real","fake","with"]
    edge_type_list = ["none","sobel","laplacian","scharr","canny"]
    img_num_list = [10,50]
    for rfw in real_fake_with_list:
        for e in edge_type_list:
            for n in img_num_list:
                main(rfw,e,n)
