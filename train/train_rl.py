
from common_import import *

print(tf.__version__)
import autokeras as ak
print(ak.__version__)




print(f"START PROGRAM: {datetime.datetime.now()}")

###パラメータ###
max_model_size = 1000000000
max_trials = 1

# model_structure = "Xception"
epochs = 50
gpu_count = 16
batch_size_per_gpu = 32
batch_size = batch_size_per_gpu * gpu_count
validation_rate = 0.1
test_rate = 0.1
cp_period = 1
data_dir = '/hss/gaisp/morilab/toshi/fake_detection/data'
classes = ['Celeb-real-image-face-90', 'Celeb-synthesis-image-face-90']
image_size = (256, 256, 3)
es_flg = False

###edge###
edge=None
# edge='sobel'
# edge='laplacian'
# edge='scharr'
# edge='canny'

###data augmentation###
# rotation_range=15.0
# width_shift_range=0.15
# height_shift_range=0.15
# brightness_range = None
# shear_range=0.0
# zoom_range=0.1
# channel_shift_range = 0.0
# horizontal_flip=True
# vertical_flip=False
###data augmentation (初期値)###
rotation_range=0.0
width_shift_range=0.0
height_shift_range=0.0
brightness_range = None
shear_range=0.0
zoom_range=0.0
channel_shift_range = 0.0
horizontal_flip=False
vertical_flip=False



###dataset作成###
print(f"START CREATE DATASET: {datetime.datetime.now()}")
train_dataset, validation_dataset, test_dataset, class_file_num, class_weights = makeImageDataGenerator_Celeb_ForAk(
        data_dir,
        classes,
        validation_rate,
        test_rate,
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
        vertical_flip,
        edge
)
# def make_gen_callable(_gen):
#     def gen():
#         for x,y in _gen:
#                 yield x,y
#     return gen
# # for i in train_generator:
# #     print(len(i))
# #     print(i[0].shape)
# #     print(i[1].shape)
# #     exit()
# train_dataset = tf.data.Dataset.from_generator(
#     # train_generator,
#     # lambda: train_generator,
#     lambda: map(np.array, train_generator),
#     # make_gen_callable(train_generator),
#     output_types=(tf.float32, tf.float32),
# )#.batch(batch_size)
# validation_dataset = tf.data.Dataset.from_generator(
#     # validation_generator,
#     # lambda: validation_generator,
#     lambda: map(np.array, validation_generator),
#     # make_gen_callable(validation_generator),
#     output_types=(tf.float32, tf.float32),
# )#.batch(batch_size)
# test_dataset = tf.data.Dataset.from_generator(
#     # test_generator,
#     # lambda: test_generator,
#     lambda: map(np.array, test_generator),
#     # make_gen_callable(test_generator),
#     output_types=(tf.float32, tf.float32),
# )#.batch(batch_size)
print(f"FINISH CREATE DATASET: {datetime.datetime.now()}")


###ディレクトリ作成###
t = time.strftime("%Y%m%d-%H%M%S")
project_name = f'AutoKeras_{t}_epoch{epochs}'
model_dir = f'../model/{project_name}'
os.makedirs(model_dir, exist_ok=True)
cp_dir = f'../model/{project_name}/checkpoint'
os.makedirs(cp_dir, exist_ok=True)
print(f"CREATE DIRECTORY: '{project_name}'")



###パラメータ保存###
params = {}
# params["model_structure"] = model_structure
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
params["edge"] = edge
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
    filepath=cp_dir+"/cp_model_{epoch:03d}+000-{accuracy:.2f}.h5",
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


###条件出力###
print("\tGPU COUNT: " + str(gpu_count))
print("\tBATCH SIZE: " + str(batch_size))
print("\tVALIDATION RATE: " + str(validation_rate))
print("\tTEST RATE: " + str(test_rate))
print("\tTRAIN DATA NUM: " + str(len(train_dataset._input_dataset)))
print("\tVALIDATION DATA NUM: " + str(len(validation_dataset._input_dataset)))
print("\tTEST DATA NUM: " + str(len(test_dataset._input_dataset)))
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
print("\tEDGE: " + str(edge))
print("")



###インスタンス生成###
clf = ak.ImageClassifier(
    project_name=project_name,
    directory=model_dir,
    loss=None,
    metrics=getMetrics("all"),
    overwrite=True,
    seed=1,
    max_model_size=max_model_size,
    max_trials=max_trials,
)



###終了時の処理###
def test_and_save_and_graph():
    global clf
    global model_dir
    global history
    global test_dataset
    testModelForAk(clf,test_dataset)
    saveModelForAk(clf,model_dir)
    makeGraph(history.history,model_dir)


###学習###
print(f"START TRAINING: {datetime.datetime.now()}")
signal.signal(signal.SIGINT, test_and_save_and_graph)
try:
    history = clf.fit(
        x = train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        class_weight=class_weights,
        verbose=1,
        workers=8,
        use_multiprocessing=False,
        callbacks=cb_list
    )
except Exception as e:
    print(e)
print(f"FINISH TRAINING: {datetime.datetime.now()}")

# モデル出力
model = clf.export_model()
model.summary()

testModelForAk(clf,test_dataset)
saveModelForAk(clf,model_dir)
makeGraph(history.history,model_dir)

print(f"FINISH PROGRAM: {datetime.datetime.now()}")














# ###学習###
# print(f"START TRAINING: {datetime.datetime.now()}")
# history = clf.fit(
#     x = train_dataset,
#     epochs=epochs,
#     validation_data=validation_dataset,
#     callbacks=cb_list
# )
# print(f"FINISH TRAINING: {datetime.datetime.now()}")



# # モデル出力
# model = clf.export_model()
# model.summary()


# testModel(model,test_dataset)
# saveModel(model,model_dir)

# print(f"FINISH PROGRAM: {datetime.datetime.now()}")



















