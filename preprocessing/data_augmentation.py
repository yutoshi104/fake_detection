
#Data Augmentation
from pathlib import Path
import cv2
import re
import numpy as np
import time
import os
import random



def data_augmentation(openfile,savefile,params):
    open_folder = Path(openfile)
    save_folder = Path(savefile)
    save_folder.mkdir(exist_ok=True)
    if ("extension" in params):
        path_list = open_folder.glob(params['extension'])
    else:
        path_list = open_folder.glob("*")

    src_file_num = 0
    for x in path_list:
        src_file_num += 1

    if ("extension" in params):
        path_list = open_folder.glob(params['extension'])
    else:
        path_list = open_folder.glob("*")

    for i,f in enumerate(path_list):
        if i==0:
            start = time.time()
            print(f"Start")
        img = cv2.imread(str(f))
        height,width,ch = img.shape
        #basename,extensionの取得
        basename = os.path.splitext(os.path.basename(f))[0]
        extension = os.path.splitext(os.path.basename(f))[1]


        # リサイズ
        if params['resize_flg']==True:
            width_ratio = width / params['resize_width']
            height_ratio = height / params['resize_height']
            if params['fill_flg']==False:
                #縦、横どちらかに揃えて、余分な部分は切り取る
                if width_ratio >= height_ratio: #横長写真
                    img = cv2.resize(img,(int(width/height_ratio),int(height/height_ratio)))
                    img = img[:, int((int(width/height_ratio)-params['resize_width'])/2):int((int(width/height_ratio)-params['resize_width'])/2)+params['resize_width'], :]
                else:   #縦長写真
                    img = cv2.resize(img,(int(width/width_ratio),int(height/width_ratio)))
                    img = img[int((int(height/width_ratio)-params['resize_height'])/2):int((int(height/width_ratio)-params['resize_height'])/2)+params['resize_height'], :, :]
            else:
                #縦、横どちらかに揃えて小さくする
                if width_ratio >= height_ratio: #横長写真
                    fill=0
                    img = cv2.resize(img,(int(width/width_ratio),int(height/width_ratio)))
                else:   #縦長写真
                    fill=1
                    img = cv2.resize(img,(int(width/height_ratio),int(height/height_ratio)))
                #写真の上下もしくは左右を埋める関数の呼び出し
                img = fill_around(img,fill,params['resize_width'],params['resize_height'],fill_color=params['fill_color'])


        #写真のバリエーションを増やす
        img_array_list = list()
        #元の画像→[0]
        img_array_list.append(img)
        #水平方向反転
        if params['reverse_flg']==True:
            img_array_list.append(horizontal_flip(img))
        #ランダムに切り抜き(これまでの画像に対して)
        if params['crop_flg']==True:
            for a in range(params['crop_num']):
                for b in range(len(img_array_list)):
                    img_array_list.append(random_crop(img_array_list[b], crop_rate_range=params['crop_rate_range']))
        #ランダムに回転(これまでの画像に対して)
        if params['rotate_flg']==True:
            l = len(img_array_list)
            for a in range(params['rotate_num']):
                for b in range(len(img_array_list)):
                    img_array_list.append(random_rotation(img_array_list[b], angle_range=params['rotate_angle_range']))



        for j,a in enumerate(img_array_list):
            save_path = save_folder.joinpath(f"{basename}_{str(j).zfill(4)}{extension}")
            cv2.imwrite(str(save_path),a)
        
        if i==0:
            print(f"Number of source files: ({src_file_num} files)")
            dst_file_num = src_file_num * (2 if params['reverse_flg']==True else 1) * (params['crop_num']+1 if params['crop_flg']==True else 1) * (params['rotate_num']+1 if params['rotate_flg']==True else 1)
            print(f"Number of augment files: ({dst_file_num} files)")
            end = time.time()
            ex_time = (end-start)*len(os.listdir(openfile))
            if ex_time<60:
                print(f"Expected time: ({int(ex_time)} seconds)")
            elif ex_time<3600:
                print(f"Expected time: ({int(ex_time/60)} minutes)")
            else:
                print(f"Expected time: ({int(ex_time/60/60)} hours)")
        print(f"Img{i+1} is complete.")
    print("End")


def fill_around(img,fill,resize_width,resize_height,fill_color="black"):
    height,width,ch = img.shape
    if fill_color=="white":
        array = np.random.randint(255,256,(resize_height,resize_width,ch),np.uint8) #白画像で埋める
    elif fill_color=="black":
        array = np.random.randint(0,1,(resize_height,resize_width,ch),np.uint8)     #黒画像で埋める
    elif fill_color=="random":
        array = np.random.randint(0,256,(resize_height,resize_width,ch),np.uint8)   #乱数画像で埋める
    else:
        array = np.random.randint(0,1,(resize_height,resize_width,ch),np.uint8)     #黒画像で埋める

    if fill:    #左右を埋める
        if random.randint(0,1): #ランダムで左右に合わせる
            if random.randint(0,1): #ランダムで左右を決める
                dis = resize_height-height
                array[dis:,:width,:] = np.array(img)
            else:
                dis = resize_height-height
                array[dis:,resize_width-width:,:] = np.array(img)
            return array
        left = int((resize_width-width)/2)
        right = left+width
        dis = resize_height-height
        array[dis:,left:right,:] = np.array(img)
        return array
    else:       #上下を埋める
        if random.randint(0,1): #ランダムで上下に合わせる
            if random.randint(0,1): #ランダムで上下を決める
                dis = resize_width-width
                array[:height,dis:,:] = np.array(img)
            else:
                dis = resize_width-width
                array[resize_height-height:,:,:] = np.array(img)
            return array
        top = int((resize_height-height)/2)
        under = top+height
        dis = resize_width-width
        array[top:under,dis:,:] = np.array(img)
        return array


# def img_augment(img,cropsize,anglerange):   #imgはnumpy配列
#     img_array_list = list()
#     #元の画像→[0]
#     img_array_list.append(img)
#     #水平方向反転→[1]
#     img_array_list.append(horizontal_flip(img))
#     #ランダムに切り抜き([0],[1]に対して)
#     for i in range(1):
#         img_array_list.append(random_crop(img_array_list[0], crop_size=cropsize))
#         img_array_list.append(random_crop(img_array_list[1], crop_size=cropsize))
#     #ランダムに回転
#     l = len(img_array_list)
#     for a in range(2):
#         for b in range(l):
#             img_array_list.append(random_rotation(img_array_list[b], angle_range=anglerange))
    
#     img_array_list.pop(0)
#     return img_array_list

def horizontal_flip(image, rate=0.5):
    #if np.random.rand() < rate: #50%の確率で実行
    image = image[:, ::-1, :]
    return image

def random_crop(image,crop_rate_range=(0.7, 1.0)):
    crop_rate = random.random()*(crop_rate_range[1]-crop_rate_range[0])+crop_rate_range[0]
    h, w, _ = image.shape
    crop_size = (int(h*crop_rate), int(w*crop_rate))

    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    croped_img = image[top:bottom, left:right, :]
    croped_img = cv2.resize(croped_img,(w,h))
    return croped_img

def random_rotation(image, angle_range=(0, 180)):
    h, w, _ = image.shape
    angle = np.random.randint(*angle_range)
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=angle, scale=1.2)
    rotated_img = cv2.warpAffine(image, M, dsize=(w, h))
    return rotated_img


if __name__=='__main__':
    #ファイルパス
    openfile="../data/Celeb-synthesis-image"
    savefile="../data/Celeb-synthesis-image-augmented"

    # コメントの先頭がデフォルト
    params = {
        # 拡張子：String(default->"*")
        "extension": "*.jpg",
        # リサイズするか：False or True
        "resize_flg": True,
            # リサイズ幅：Int
            "resize_width": 640,
            # リサイズ高さ：Int
            "resize_height": 480,
            # 埋めるか(Falseなら切り取り)：False or True
            "fill_flg": False,
                # 埋める色："black" or "white" or "noize"
                "fill_color": "black",
        # 水平反転するか：False or True
        "reverse_flg": True,
        # ランダムに切り取るか：False or True
        "crop_flg": True,
            # ランダム切り取りの縮小率の範囲：Tuple
            "crop_rate_range": (0.85,1.0),
            # ランダム切り取り回数：Int
            "crop_num": 1,
        # ランダムに回転するか：False or True
        "rotate_flg": False,
            # ランダム回転角度の範囲：Tuple
            "rotate_angle_range": (-25,25),
            # ランダム回転の回数：Int
            "rotate_num": 2,
    }
    data_augmentation(openfile,savefile,params)
