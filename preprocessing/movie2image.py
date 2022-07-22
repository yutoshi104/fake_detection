import cv2
import os
import glob


def save_all_frames(video_path, dir_path, basename, step=1, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if (n%step)==0:
                cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            return



if __name__=='__main__':
    video_directory = "../data/Celeb-DF-v2/Celeb-synthesis/*"
    image_directory = "../data/datas/Celeb-synthesis-image"
    # video_directory = "../data/Celeb-DF-v2/Celeb-real/*"
    # image_directory = "../data/datas/Celeb-real-image"
    # video_directory = "/data/toshikawa/DFMNIST+/fake_dataset/*/*.mp4"
    # image_directory = "/data/toshikawa/datas/DFMNIST-fake-image"
    # video_directory = "/data/toshikawa/DFMNIST+/real_dataset/*/*.mp4"
    # image_directory = "/data/toshikawa/datas/DFMNIST-real-image"
    step = 1    #何フレームごとに画像を保存するか

    videos = sorted(glob.glob(video_directory))
    for video in videos:
        basename = os.path.splitext(os.path.basename(video))[0]
        print(basename)
        save_all_frames(video, image_directory, basename, step)
