from PIL import Image
from PIL.ExifTags import TAGS
import glob
import pyexiv2

def get_exif_of_image(file):
    """Get EXIF of an image if exists.

    指定した画像のEXIFデータを取り出す関数
    @return exif_table Exif データを格納した辞書
    """

    img = pyexiv2.Image(file)
    print(path,end=" ")
    metadata = img.read_exif()
    return metadata



    # im = Image.open(file)

    # # Exif データを取得
    # # 存在しなければそのまま終了 空の辞書を返す
    # try:
    #     exif = im._getexif()
    # except AttributeError:
    #     return {}
    # if exif==None:
    #     return {}

    # # タグIDそのままでは人が読めないのでデコードして
    # # テーブルに格納する
    # exif_table = {}
    # for tag_id, value in exif.items():
    #     tag = TAGS.get(tag_id, tag_id)
    #     exif_table[tag] = value

    # return exif_table



if __name__=='__main__':
    data_dir = '../data/datas'
    classes = ['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']

    for l,c in enumerate(classes):
        print(c,end=" ")
        image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
        print(len(image_path_list))
        for path in image_path_list:
            print(path,end=" ")
            a = get_exif_of_image(path)
            if a is not {}:
                print(a)