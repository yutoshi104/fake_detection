from pathlib import Path
from PIL import Image # Pillow ライブラリ
import os
import glob


def checkBreakImage(path):
    # 画像が何かしらの理由で壊れていたらTrueを返す。

    image_file = Path(path)
    with image_file.open('rb') as f:
        try:
            im = Image.open(f, 'r')
        # except UnidentifiedImageError as e:
        except Exception as e:
            print(f'NG1 {image_file.name} {e.__class__.__name__}')
            return True

        try:
            im.verify()
        except Exception as e:
            print(f'NG2 {image_file.name} {e.__class__.__name__}')
            return True


    return False



if __name__=='__main__':
    data_dir = '/hss/gaisp/morilab/toshi/fake_detection/data'
    classes = ['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']
    delete_flg = False

    for l,c in enumerate(classes):
        print(c,end=": ")
        image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
        print(len(image_path_list))
        for path in image_path_list:
            res = checkBreakImage(path)
            if res:
                print(path,"is broken.",end=" ")
                if delete_flg:
                    os.remove(path)
                    print("removed.")
                else:
                    print()
