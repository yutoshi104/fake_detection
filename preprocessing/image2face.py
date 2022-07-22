# GPU無効化
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

# 単一GPUデバイス指定
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf

# # 複数GPU指定
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Create 2 virtual GPUs with 1GB memory each
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11264),tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11264)]
#         )
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)


from tensorflow.python.client import device_lib
tf.get_logger().setLevel("ERROR")
from mtcnn.mtcnn import MTCNN
print(tf.__version__)
print(device_lib.list_local_devices())
import cv2
import os
import glob
import copy
from concurrent import futures
import time

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    detector = MTCNN()


def image2face(src_path, dst_directory, del_flg=False, square_size=None, padding_rate=0, confidence_threshold=0.9, biggest_confidence=True, ext='jpg', index=0):
    print(str(index)+"/"+str(file_num)+"\t"+image)

    # 読み込み
    if not os.path.exists(src_path):
        print(f'file "{src_path}" is not exist.')
        return
    src_data = cv2.imread(src_path)
    h,w,c = src_data.shape
    pixels = cv2.cvtColor(src_data, cv2.COLOR_BGR2RGB)

    # 顔抽出
    # try:
    faces = detector.detect_faces(pixels)
    # except:
    #     with open("error.log",mode="w") as f:
    #         f.write(src_path+"\n")
    #     return

    # パス作成
    os.makedirs(dst_directory, exist_ok=True)
    basename = os.path.splitext(os.path.basename(src_path))[0]
    dst_path = os.path.join(dst_directory, basename+"."+ext)

    max_confidence = -1
    # そもそも顔が検出されなければスキップ
    if len(faces)==0:
        print(f"\tskip: face = 0")
        return
    for i in range(len(faces)):
        # 閾値よりも低い信頼度であればスキップ
        confidence = faces[i]['confidence']
        if confidence < confidence_threshold:
            print(f"\tskip:「{src_path}-{i}」(confidence:{confidence})")
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

        if biggest_confidence:
            if max_confidence < confidence:
                max_confidence = confidence
                md = croped_data
        else:
            cv2.imwrite(dst_path, croped_data)

    # 最大信頼度の時のみ保存
    if biggest_confidence and max_confidence > 0:
        cv2.imwrite(dst_path, md)
        # if del_flg and os.path.exists(src_path):
        #     os.remove(src_path)

    print("fin:"+str(index))




if __name__=='__main__':
    ###パラメータ###
    # src_directory = "../data/datas/Celeb-real-image"
    # # dst_directory = "../data/datas/Celeb-real-image-face-90"
    # dst_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-real-image-face-90"
    src_directory = "../data/datas/Celeb-synthesis-image"
    # dst_directory = "../data/datas/Celeb-synthesis-image-face-90"
    dst_directory = "/hss/gaisp/morilab/toshi/fake_detection/data/Celeb-synthesis-image-face-90"
    del_flg = False
    square_size = 256
    padding_rate = 0.1
    confidence_threshold = 0.90

    images = sorted(glob.glob(src_directory+"/*"))  # 本来はこれだけ
    print("all file num: "+str(len(images)))
    # images_base = copy.deepcopy(images)
    # for i in range(len(images)):
    #     images_base[i] = os.path.basename(images[i])
    # dst_images = sorted(glob.glob(dst_directory+"/*"))
    # dst_images_base = copy.deepcopy(dst_images)
    # for i in range(len(dst_images)):
    #     dst_images_base[i] = os.path.basename(dst_images[i])
    # l = set(images_base) - set(dst_images_base)
    # l = list(l)
    # _images = copy.deepcopy(images)
    # images = []
    # for i in range(len(_images)):
    #     basename = os.path.basename(_images[i])
    #     if basename in l:
    #         images.append(_images[i])
    # del images_base
    # del dst_images
    # del dst_images_base
    # del _images
    # del l
    dst_images = sorted(glob.glob(dst_directory+"/*"))
    print(len(images))
    print(len(dst_images))
    for i in range(len(dst_images)):
        dst_images[i] = src_directory+"/"+os.path.basename(dst_images[i])
    images = set(images) - set(dst_images)
    images = list(images)
    images.sort()
    print("image_num: "+str(len(images)))
    del dst_images
        
    
    # スタート位置
    start = 0 # 0〜max_split-1
    max_split = 16
    l = len(images)
    position = round(l * (start / max_split))
    images = images[position:] + images[:position]
    print("max_split: "+str(max_split))
    print("start: "+str(start))

    start = time.time()

    # future_list = []
    # file_num = len(images)
    # with futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     for i,image in enumerate(images):
    #         future = executor.submit(
    #             fn=image2face,
    #             src_path=image,
    #             dst_directory=dst_directory,
    #             square_size=square_size,
    #             padding_rate=padding_rate,
    #             confidence_threshold=confidence_threshold,
    #             index=i+1
    #         )
    #         # future_list.append(future)
    #         # if i>100:
    #         #     break
    #     # _ = futures.as_completed(fs=future_list)


    file_num = len(images)
    print("file_num: "+str(file_num))
    for i,image in enumerate(images):
        # print(image)

        image2face(
            image,
            dst_directory,
            del_flg=del_flg,
            square_size=square_size,
            padding_rate=padding_rate,
            confidence_threshold=confidence_threshold,
            index=i+1
        )

        # if i>100:
        #     break

    

    # 処理時間表示
    elapsed_time = time.time() - start
    print(f"\telapsed_time: {elapsed_time}[sec]")
