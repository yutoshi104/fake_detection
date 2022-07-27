####################################################################################################
###実行コマンド###
# nohup python train.py > train_out.log &
###強制終了コマンド###
# jobs      (実行中のジョブ一覧)
# kill 1    (1のジョブを強制終了)
# fg 1      (1をフォアグランド実行に戻す)
# ctrl+c    (強制終了)
    # ctrl+z    (停止)
    # bg %1     (1をバックグラウンド実行で再開)
####################################################################################################

from common_import import *

print(f"START PROGRAM: {datetime.datetime.now()}")

###パラメータ###
# model_structure = "SampleCnn"
# model_structure = "Vgg16"
# model_structure = "InceptionV3"
# model_structure = "Xception"
# model_structure = "EfficientNetV2"
model_structure = "OriginalNet"
# model_structure = "OriginalNetNonDrop"
epochs = 100
gpu_count = 8
batch_size_per_gpu = 32
batch_size = batch_size_per_gpu * gpu_count
validation_rate = 0.1
test_rate = 0.1
# ↑動画ごとに分けているので最終的な画像でのデータ数はだいたい...
cp_period = 5
# data_dir = '../data/datas'
data_dir = '/hss/gaisp/morilab/toshi/fake_detection/data'
# classes = ['yuto', 'b']
# classes = ['Celeb-real-image', 'Celeb-synthesis-image']
classes = ['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']
# image_size = (480, 640, 3)
# image_size = (240, 320, 3)
image_size = (256, 256, 3)
# image_size = (128, 128, 3)
es_flg = False


###data augmentation###
rotation_range=15.0
width_shift_range=0.15
height_shift_range=0.15
brightness_range = None
shear_range=0.0
zoom_range=0.1
channel_shift_range = 0.0
horizontal_flip=True
vertical_flip=False
###data augmentation (初期値)###
# rotation_range=0.0
# width_shift_range=0.0
# height_shift_range=0.0
# brightness_range = None
# shear_range=0.0
# zoom_range=0.0
# channel_shift_range = 0.0
# horizontal_flip=False
# vertical_flip=False


###モデルの生成###
model = globals()['load'+model_structure](input_shape=image_size)
# model.summary()


###Generator作成###
print(f"START CREATE GENERATOR: {datetime.datetime.now()}")
class_file_num = {}
class_weights = {}
train_data = []
validation_data = []
test_data = []
train_rate = 1 - validation_rate - test_rate
s1 = (int)(59*train_rate)
s2 = (int)(59*(train_rate+validation_rate))
id_list = list(range(62))
id_list.remove(13)
id_list.remove(14)
id_list.remove(18)
# random.shuffle(id_list)
train_id_list = id_list[ : s1]
validation_id_list = id_list[s1 : s2]
test_id_list = id_list[s2 : ]
print("\tTRAIN IMAGE DATA ID: ",end="")
print(train_id_list)
print("\tVALIDATION IMAGE DATA ID: ",end="")
print(validation_id_list)
print("\tTEST IMAGE DATA ID: ",end="")
print(test_id_list)
del id_list
data_num = 0
for l,c in enumerate(classes):
    image_path_list = sorted(glob.glob(data_dir+"/"+c+"/*"))
    path_num = len(image_path_list)
    data_num += path_num
    regexp = r'^.+?id(?P<id>(\d+))(_id(?P<id2>\d+))?_(?P<key>\d+)_(?P<num>\d+).(?P<ext>.{2,4})$'
    past_path = image_path_list[0]
    movie_image_list = []
    for i in range(1,len(image_path_list)):
        past_ids = re.search(regexp,past_path).groupdict()
        now_ids = re.search(regexp,image_path_list[i]).groupdict()
        if (past_ids['id']==now_ids['id']) and (past_ids['id2']==None or past_ids['id2']==now_ids['id2']) and (past_ids['key']==now_ids['key']):
            movie_image_list.append([image_path_list[i],l])
        else:
            if int(past_ids['id']) in train_id_list:
                train_data.append(movie_image_list)
            elif int(past_ids['id']) in validation_id_list:
                validation_data.append(movie_image_list)
            elif int(past_ids['id']) in test_id_list:
                test_data.append(movie_image_list)
            movie_image_list = []
            movie_image_list.append([image_path_list[i],l])
        past_path = image_path_list[i]
    # 不均衡データ調整
    class_file_num[c] = path_num
    if l==0:
        n = class_file_num[c]
    class_weights[l] = 1 / (class_file_num[c]/n)
print("\tMOVIE NUM: " + str(len(train_data)+len(validation_data)+len(test_data)))
# print(train_id_list)
# print(validation_id_list)
# print(test_id_list)
# with open('myresult.txt', 'w', encoding='UTF-8') as f:
#     for d in train_data:
#         for p in d:
#             f.write("%s\n" % p)
#     f.write(str(len(train_data))+"\n\n\n\n\n\n\n\n\n\n")
#     for d in validation_data:
#         for p in d:
#             f.write("%s\n" % p)
#     f.write(str(len(validation_data))+"\n\n\n\n\n\n\n\n\n\n")
#     for d in test_data:
#         for p in d:
#             f.write("%s\n" % p)
#     f.write(str(len(test_data))+"\n\n\n\n\n\n\n\n\n\n")
# f.close()
# exit()
train_data = list(chain.from_iterable(train_data))
validation_data = list(chain.from_iterable(validation_data))
test_data = list(chain.from_iterable(test_data))
train_data_num = len(train_data)
validation_data_num = len(validation_data)
test_data_num = len(test_data)
print("\tALL IMAGE DATA NUM: " + str(data_num))
print("\tTRAIN IMAGE DATA NUM: " + str(train_data_num))
print("\tVALIDATION IMAGE DATA NUM: " + str(validation_data_num))
print("\tTEST IMAGE DATA NUM: " + str(test_data_num))
def makeGenerator(data,subset="training"):
    return ImageIterator(
        data,
        batch_size=batch_size,
        target_size=image_size[:2],
        color_mode='rgb',
        seed=1,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=1./255,
        data_format='channels_last',
        subset=subset)
train_generator = makeGenerator(train_data,"training")
validation_generator = makeGenerator(validation_data,"validation")
test_generator = makeGenerator(test_data,"test")
del train_data
del validation_data
del test_data
print(f"FINISH CREATE GENERATOR: {datetime.datetime.now()}")


###ディレクトリ作成###
t = time.strftime("%Y%m%d-%H%M%S")
model_dir = f'../model/{model_structure}_{t}_epoch{epochs}'
os.makedirs(model_dir, exist_ok=True)
cp_dir = f'../model/{model_structure}_{t}_epoch{epochs}/checkpoint'
os.makedirs(cp_dir, exist_ok=True)


###パラメータ保存###
params = {}
params["model_structure"] = model_structure
params["epochs"] = epochs
params["batch_size_per_gpu"] = batch_size_per_gpu
params["validation_rate"] = validation_rate
params["test_rate"] = test_rate
params["cp_period"] = cp_period
params["data_dir"] = data_dir
params["classes"] = classes
params["image_size"] = image_size
params["es_flg"] = es_flg
params["rotation_range"] = rotation_range
params["width_shift_range"] = width_shift_range
params["height_shift_range"] = height_shift_range
params["brightness_range"] = brightness_range
params["shear_range"] = shear_range
params["zoom_range"] = zoom_range
params["channel_shift_range"] = channel_shift_range
params["horizontal_flip"] = horizontal_flip
params["vertical_flip"] = vertical_flip
saveParams(params,filename=model_dir+"/params.json")


###callback作成###
cb_list = []
if es_flg:
    ##↓監視する値の変化が停止した時に訓練を終了##
    es_callback = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        verbose=1,
        mode='auto'
    )
    cb_list.append(es_callback)
cp_callback = callbacks.ModelCheckpoint(
    filepath=cp_dir+"/cp_model_{epoch:03d}-{accuracy:.2f}.h5",
    monitor='val_loss',
    mode='auto',
    save_best_only=False,
    save_weights_only=False,
    verbose=1,
    period=cp_period
)
cb_list.append(cp_callback)
sh_callback = saveHistory(filepath=model_dir+"/history.json")
cb_list.append(sh_callback)


##サンプルデータセット使用###
# train_generator,validation_generator,test_generator = getSampleData()


###条件出力###
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTRAIN DATA NUM: " + str(train_data_num))
print("\tVALIDATION DATA NUM: " + str(validation_data_num))
print("\tTEST DATA NUM: " + str(test_data_num))
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




# def test_and_save(a=None,b=None):
#     global model
#     global model_dir
#     global history
#     global test_generator

#     ###テスト###
#     print(f"START TEST: {datetime.datetime.now()}")
#     loss_and_metrics = model.evaluate_generator(
#         test_generator
#     )
#     print("Test loss:",loss_and_metrics[0])
#     print("Test accuracy:",loss_and_metrics[1])
#     print("Test AUC:",loss_and_metrics[2])
#     print("Test Precision:",loss_and_metrics[3])
#     print("Test Recall:",loss_and_metrics[4])
#     print("Test TP:",loss_and_metrics[5])
#     print("Test TN:",loss_and_metrics[6])
#     print("Test FP:",loss_and_metrics[7])
#     print("Test FN:",loss_and_metrics[8])
#     print(f"FINISH TEST: {datetime.datetime.now()}")

#     ###モデルの保存###
#     print(f"START MODEL SAVE: {datetime.datetime.now()}")
#     try:
#         model.save(f'{model_dir}/model.h5')
#     except NotImplementedError:
#         print('Error')
#     model.save_weights(f'{model_dir}/weight.hdf5')
#     print(f"FINISH MODEL SAVE: {datetime.datetime.now()}")


#     ###グラフ化###
#     try:
#         fig = plt.figure()
#         plt.plot(range(1, len(history.history['accuracy'])+1), history.history['accuracy'], "-o")
#         plt.plot(range(1, len(history.history['val_accuracy'])+1), history.history['val_accuracy'], "-o")
#         plt.title('Model accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.grid()
#         plt.legend(['accuracy','val_accuracy'], loc='best')
#         fig.savefig(model_dir+"/result.png")
#         # plt.show()
#     except NameError:
#         print("The graph could not be saved because the process was interrupted.")

###終了時の処理###
def test_and_save_and_graph():
    global model
    global model_dir
    global history
    global test_generator
    testModel(model,test_generator)
    saveModel(model,model_dir)
    makeGraph(history.history,model_dir)


###学習###
print(f"START TRAINING: {datetime.datetime.now()}")
signal.signal(signal.SIGINT, test_and_save_and_graph)
try:
    history = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        # steps_per_epoch=10,
        epochs=epochs,
        class_weight=class_weights,
        verbose=1,
        workers=8,
        use_multiprocessing=False,
        callbacks=cb_list
    )
except Exception as e:
    print(e)
print(f"FINISH TRAINING: {datetime.datetime.now()}")

testModel(model,test_generator)
saveModel(model,model_dir)
makeGraph(history.history,model_dir)

print(f"FINISH PROGRAM: {datetime.datetime.now()}")
