import cv2
import sys
import os
import glob
import json
import csv
import time
import copy
import random
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint


### ログ表示 ###
def print_log(message="", line_break=True):
    if line_break:
        sys.stdout.write(str(message) + "\n")
    else:
        sys.stdout.write(str(message))
    sys.stdout.flush()


import tensorflow as tf
from tensorflow.python.client import device_lib
tf.get_logger().setLevel("ERROR")
from mtcnn.mtcnn import MTCNN
print_log(tf.__version__)
print_log(device_lib.list_local_devices())


core_num = multiprocessing.cpu_count()
print_log(f"CPU Core: {core_num}")

# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
detector = MTCNN()



def image2face(src_data, square_size=256, padding_rate=0.1, confidence_threshold=0.9, ext='jpg', basename="image"):
    ##################################################
    # MTCNNで抽出した顔のうち最大精度のものとその精度を返す
    ##################################################

    # 読み込み
    h,w,c = src_data.shape
    pixels = cv2.cvtColor(src_data, cv2.COLOR_BGR2RGB)

    # 顔抽出
    faces = detector.detect_faces(pixels)

    max_confidence = -1
    # そもそも顔が検出されなければスキップ
    if len(faces)==0:
        print_log(f"\t\tskip: no face")
        return None
    for i in range(len(faces)):
        # 閾値よりも低い信頼度であればスキップ
        confidence = faces[i]['confidence']
        if confidence < confidence_threshold:
            print_log(f"\t\tskip:「{basename}-{i}」(confidence:{confidence})")
            continue

        x1, y1, width, height = faces[i]['box']

        # 正方形切り取り(正方形で切り取れる部分しか残せない)
        if square_size:
            if width > height:
                length = width
                diff = width-height
                y1 -= int(diff/2)
                height = width
                # はみ出していたら
                if y1 < 0:
                    y1 = 0
                if w < width:
                    height = h
                    width = h
            else:
                length = height
                diff = height-width
                x1 -= int(diff/2)
                width = height
                # はみ出していたら
                if x1 < 0:
                    x1 = 0
                if w < width:
                    width = w
                    height = w  

        # パディング(上下左右それぞれに、padding_rate分の、上下左右で平等なパディングを施す)
        if padding_rate:
            min_p = int(length*padding_rate)
            if x1-min_p < 0 and x1 < min_p:
                min_p = x1
            if x1+width+min_p > w and w-(x1+width) < min_p:
                min_p = w-(x1+width)
            if y1-min_p < 0 and y1 < min_p:
                min_p = y1
            if y1+height+min_p > h and h-(y1+height) < min_p:
                min_p = h-(y1+height)
            if min_p < 0:
                min_p = 0
            x1 -= min_p
            y1 -= min_p
            width += min_p*2
            height += min_p*2

        x2, y2 = x1+width, y1+height
        croped_data = src_data[y1:y2, x1:x2]

        # 正方形ならリサイズ
        if square_size:
            croped_data = cv2.resize(croped_data, dsize=(square_size,square_size))

        if max_confidence < confidence:
            max_confidence = confidence
            md = croped_data

    # 最大信頼度の時のみ保存
    if max_confidence > 0:
        print_log("\t\tfin:")
        return (md, max_confidence)
    else:
        print_log("\t\tNot found max confidence.")
        return None




def save_all_frames(video_path, save_dir, label, step=1, square_size=256, padding_rate=0.1, confidence_threshold=0.9, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    basename_without_extension = os.path.splitext(os.path.basename(video_path))[0]

    # 既に変換していた場合はスキップ
    already_list = get_processed_videos()
    if basename_without_extension in already_list:
        return

    os.makedirs(save_dir, exist_ok=True)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if (n%step)==0:
                print_log(f'\tframe {n}')
                # 顔抽出処理
                result = image2face(frame, square_size=square_size, padding_rate=padding_rate, confidence_threshold=confidence_threshold, ext=ext, basename=f'{basename_without_extension}_{str(n).zfill(digit)}')
                if result is not None:
                    (img, confidence) = result
                    # cv2.imwrite(f'{save_dir}/{label}_{basename_without_extension}_{str(n).zfill(digit)}_{str(int(confidence*1000)).zfill(5)}.{ext}', img)
                    cv2.imwrite(f'{save_dir}/{label}_{basename_without_extension}_{str(n).zfill(digit)}.{ext}', img)
            n += 1
        else:
            # 終了証明
            with open('./logs_dfdc/fin.txt', encoding='UTF-8', mode='a') as f:
                f.write(basename_without_extension+'\n')
            return



def get_processed_videos(filepath='./logs_dfdc/fin.txt'):
    with open(filepath, encoding='UTF-8', mode='r') as f:
        return [s.rstrip() for s in f.readlines()]






if __name__=='__main__':

    label_directory = "/hss/gaisp/morilab/toshi/fake_detection/dfdc/"
    video_directory = "/hss/gaisp/morilab/toshi/fake_detection/dfdc_mp4/"
    save_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/dfdc-face-90/"
    variation = ["train","validation","test"]
    meta_extension = ['.json','.csv','.json']
    step = 1 #何フレームごとに画像を保存するか
    square_size = 256 #最終的な画像サイズ
    padding_rate = 0.1 #パディングの割合
    confidence_threshold = 0.90 #顔抽出する最低の信頼精度(この信頼精度以上ないと顔抽出しない)
    ext = 'jpg' #保存拡張子


    # path_listやlabelの準備
    for vi,v in enumerate(variation):
        print_log(f"Convert {v} data.")
        video_paths = sorted(glob.glob(video_directory+v+"/"+"*.mp4"))
        processed_videos = get_processed_videos()
        unprocessed_video_paths = [vp for vp in video_paths if os.path.splitext(os.path.basename(vp))[0] not in processed_videos]
        
        label_paths = []
        for root, dirs, files in os.walk(top=label_directory+v):
            for file in files:
                if not file.lower().endswith((meta_extension[vi])):
                    continue
                json_path = os.path.join(root, file)
                label_paths.append(json_path)
        label_paths = sorted(label_paths)
        labels = {}
        if meta_extension[vi] == '.json':
            for lp in label_paths:
                with open(lp, encoding='utf8', newline='') as f:
                    meta_json = json.load(f)
                    labels.update(meta_json)
        elif meta_extension[vi] == '.csv':
            for lp in label_paths:
                with open(lp, encoding='utf8', newline='') as f:
                    csvreader = csv.reader(f)
                    for row in csvreader:
                        if row[0]=='filename' and row[1]=='label':
                            continue
                        labels[row[0]] = {
                            "is_fake": int(row[1]),
                        }
        
        # 画像化 & 抽出
        random.shuffle(unprocessed_video_paths)
        print_log(f"unprocessed_video_paths_num: {len(unprocessed_video_paths)}")
        # with ThreadPoolExecutor(max_workers=core_num) as executor:
        for index,vp in enumerate(unprocessed_video_paths):
            print_log(f"'{vp}': ")
            basename = os.path.basename(vp)
            if basename not in labels:
                print_log(f"\tno label.")
                continue
            if 'label' in labels[basename]:
                label = 0 if labels[basename]['label']=="REAL" else 1
            elif 'is_fake' in labels[basename]:
                label = int(labels[basename]['is_fake'])
            else:
                print_log(f"\tno label.")
                continue
            save_all_frames(vp, save_directory+v, label, step, square_size, padding_rate, confidence_threshold, ext)
