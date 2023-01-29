from common_import import *

print(f"START PROGRAM: {datetime.datetime.now()}")

###パラメータ###
# retrain_dir = "OriginalNet_20220731-130612_epoch50"
# retrain_dir = "OriginalNetNonDrop_20220731-131333_epoch50"
retrain_dir = "OriginalNet_20221121-174136_epoch50"
# retrain_dir = "Xception_20220812-065159_epoch15"
retrain_epochs = 0
gpu_count = 32
# data_dir = "/hss/gaisp/morilab/toshi/fake_detection/data"
data_dir = "/home/toshi/fake_detection/data/datas"
# classes = ['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']
# classes = ['Celeb-real-image-face-90-blur', 'Celeb-synthesis-image-face-90-blur']
# classes = ['Celeb-real-image-face-90']
# classes = ['Celeb-synthesis-image-face-90']
# classes = ['Celeb-real-image-face-90-blur']
classes = ['Celeb-synthesis-image-face-90-blur']
image_size = (256, 256, 3)


###パラメータ読込###
model_dir = "/home/toshi/fake_detection/model/"
params = loadParams(model_dir+retrain_dir+"/params.json")
model_structure = params['model_structure']
epochs = params['epochs']
batch_size_per_gpu = params['batch_size_per_gpu']
batch_size = batch_size_per_gpu * gpu_count
validation_rate = params['validation_rate']
test_rate = params['test_rate']
cp_period = params['cp_period']
# data_dir = params['data_dir']
# classes = params['classes']
# image_size = tuple(params['image_size'])
es_flg = params['es_flg']
rotation_range=params['rotation_range']
width_shift_range=params['width_shift_range']
height_shift_range=params['height_shift_range']
brightness_range=params['brightness_range']
shear_range=params['shear_range']
zoom_range=params['zoom_range']
channel_shift_range=params['channel_shift_range']
horizontal_flip=params['horizontal_flip']
vertical_flip=params['vertical_flip']


###modelパス取得###
if 0 == len(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')):
    print(retrain_dir+" has no checkpoint.")
    exit()
exist_learned_model = True if 0!=len(glob.glob(model_dir+retrain_dir+'/model.h5')) else False

newest_checkpoint_path = list(sorted(glob.glob(model_dir+retrain_dir+'/checkpoint/*.h5')))[-1]
# expect_epochs = int(re.findall(r'^.+epoch_(\d+).+?$', retrain_dir)[0])
expect_epochs = epochs
e = re.findall(r'^cp_model_(\d+)(\+\d+)?-.+?\.h5$', os.path.basename(newest_checkpoint_path))
if e[0][1]=='':
    saved_epochs = int(e[0][0])
else:
    saved_epochs = int(e[0][0]) + int(e[0][1])

model_path = model_dir+retrain_dir+'/model.h5' if exist_learned_model else newest_checkpoint_path

print("model path: ",end="")
print(model_path)
print("saved epochs: ",end="")
print(saved_epochs)


###モデルの読み込み###
model = globals()['load'+model_structure](input_shape=image_size,weights_path=model_path)
model.summary()


###Generator作成###
print(f"START CREATE GENERATOR: {datetime.datetime.now()}")
# _, _, test_generator, class_file_num, class_weights = makeImageDataGenerator_Celeb(
#     data_dir,
#     classes,
#     validation_rate,
#     test_rate,
#     batch_size,
#     image_size,
#     rotation_range,
#     width_shift_range,
#     height_shift_range,
#     brightness_range,
#     shear_range,
#     zoom_range,
#     channel_shift_range,
#     horizontal_flip,
#     vertical_flip
# )
test_generator, class_file_num, class_weights = makeImageDataGenerator(
    data_dir,
    classes,
    batch_size,
    image_size,
    rotation_range,
    width_shift_range,
    height_shift_range,
    brightness_range,
    shear_range,
    zoom_range,
    channel_shift_range,
    horizontal_flip,
    vertical_flip
)
print(f"FINISH CREATE GENERATOR: {datetime.datetime.now()}")


###条件出力###
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTEST DATA NUM: " + str(test_generator.len()))
print("\tCHECKPOINT PERIOD: " + str(cp_period))
print("\tDATA DIRECTORY: " + str(data_dir))
print("\tCLASSES: " + str(classes))
print("\tCLASSES NUM: " + str(class_file_num))
print("\tIMAGE SIZE: " + str(image_size))
print("\tEARLY STOPPING: " + str(es_flg))
print("\tROTATION RANGE: " + str(rotation_range))
print("\tWIDTH SHIFT RANGE: " + str(width_shift_range))
print("\tHEIGHT SHIFT RANGE: " + str(height_shift_range))
print("\tBRIGHTNESS RANGE: " + str(brightness_range))
print("\tSHEAR RANGE: " + str(shear_range))
print("\tZOOM RANGE: " + str(zoom_range))
print("\tCHANNEL SHIFT RANGE: " + str(channel_shift_range))
print("\tHORIZONTAL FLIP: " + str(horizontal_flip))
print("\tVERTICAL FLIP: " + str(vertical_flip))
print("")
print("\tTRAINED EPOCHS: " + str(saved_epochs))
print("")











testModel(model,test_generator)

